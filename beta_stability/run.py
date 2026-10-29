"""Running the algorithm a config asks for.

The config is the whole interface. It names the algorithm under
``algorithm_type``, carries every parameter that algorithm needs, and says
where results go. Four layers, each owned by one module:

1. :mod:`beta_stability.config` turns paths and command-line overrides into a
   config dict.
2. that config defines the run.
3. :func:`beta_stability.algorithm_preparation.create_algorithm` turns the
   config into a prepared algorithm plus its shared data.
4. this module runs it, and :mod:`beta_stability.data.results` writes what it
   produced.

Nothing here evaluates anything: the algorithm scores itself through the
validator it was prepared with.

CommandLine:
    python -m beta_stability --config species.yaml [--base_config base.yaml]
"""

import argparse
import logging
import os
import time

from beta_stability import config as config_module
from beta_stability.data import results
from beta_stability.edge_weights import reviewer
from beta_stability.algorithm_preparation import create_algorithm, get_initial_edges
from beta_stability.init_logger import init_logger
from beta_stability.data.io import EmptyDataframeException, SingleAnnotationException

logger = logging.getLogger("beta_stability")


def run_once(config):
    """Prepare the algorithm named by the config, run it, and write its graph.

    Returns ``(clustering, node2uuid)``.
    """
    logger.info(f"Starting {config['algorithm_type'].upper()} algorithm")
    try:
        algorithm, common_data = create_algorithm(config)
    except EmptyDataframeException as e:
        return _degenerate_result(config, e, empty=True)
    except SingleAnnotationException as e:
        return _degenerate_result(config, e, empty=False)

    output_path = common_data['output_path']
    os.makedirs(output_path, exist_ok=True)

    init_verifiers = common_data.get('init_verifiers', [common_data['verifier_name']])
    logger.info(f"Initializing graph using verifiers: {init_verifiers}")
    for name in init_verifiers:
        embeddings = common_data['embeddings_dict'][name]
        if hasattr(embeddings, 'k'):
            logger.info(f"K-means found {embeddings.k} clusters")

    initial_edges = get_initial_edges(common_data, config)
    logger.info(f"Starting with {len(initial_edges)} initial edges")
    requested_edges = algorithm.step(initial_edges)

    # An algorithm that presents one review per step declares how many steps it
    # may need (manual and thresholded review); the others keep the configured cap.
    max_outer = max(config['algorithm']['max_outer_iterations'],
                    int(getattr(algorithm, 'required_outer_iterations', 0)))
    human_reviewer = common_data.get('human_reviewer')
    waits_for_people = getattr(human_reviewer, 'waits_for_people', False)
    iteration = 0
    waiting = False
    while not algorithm.is_finished() and iteration < max_outer:
        if not waiting:
            logger.info(f"Outer iteration {iteration}")
        if requested_edges:
            responses, quit_requested = reviewer.responses(requested_edges, human_reviewer)
            if quit_requested:
                logger.info("Human reviewer requested to quit")
                break
            # People answer in their own time. Until some answer arrives the
            # graph cannot change, so poll the same request rather than spend
            # an iteration recomputing it.
            if waits_for_people and not responses:
                if not waiting:
                    logger.info(f"Waiting for reviews of {len(requested_edges)} pairs")
                    waiting = True
                time.sleep(human_reviewer.poll_seconds)
                continue
            waiting = False
            # Only the stability algorithm records "can't tell" answers (None
            # score); the others read every human score as same or different.
            if not hasattr(algorithm, 'cant_tell'):
                responses = [r for r in responses if r[2] is not None]
            requested_edges = algorithm.step(responses)
        else:
            logger.info("No edges for human review, continuing algorithm")
            requested_edges = algorithm.step([])
        iteration += 1

    if iteration >= max_outer:
        logger.warning(f"Reached max outer iterations ({max_outer})")
    else:
        logger.info(f"Algorithm converged after {iteration} outer iterations")

    withdraw_pending = getattr(human_reviewer, 'withdraw_pending', None)
    if withdraw_pending is not None:
        withdraw_pending()

    logger.info("Algorithm execution completed")
    algorithm.show_stats()

    clustering, node2cid, graph = algorithm.get_clustering()
    node2uuid = common_data['node2uuid']
    logger.info(f"Saving results to {output_path}")
    results.write_graph(config, clustering, node2cid, node2uuid, graph, output_path)
    return clustering, node2uuid


def _degenerate_result(config, exception, empty):
    """Write the graph files for a dataset too small to run on.

    Downstream tooling expects the files to exist for every run, so an empty
    dataset produces empty ones and a single annotation produces a single
    cluster, rather than nothing at all.
    """
    output_path = config['data']['output_path']
    if empty:
        logger.error(exception)
        logger.info(f"Saving empty results to {output_path}")
        results.write_graph(config, {}, {}, {}, {}, output_path)
        return {}, {}

    logger.info(exception)
    logger.info(f"Saving single-annotation result to {output_path}")
    node_id = next(iter(exception.node2uuid))
    node2uuid = {str(k): v for k, v in exception.node2uuid.items()}
    results.write_graph(config, {'0': [node_id]}, {str(node_id): '0'}, node2uuid, {},
                        output_path)
    return {'0': [node_id]}, exception.node2uuid


def run_from_config(config, save_dir=None):
    """Run everything the config asks for; returns the annotation file written.

    One run for a plain config, or one run per field-value combination when
    the config separates by fields, whose clusterings are renumbered into a
    single combined annotation file.
    """
    output_base = config['data']['output_path']
    save_dir = save_dir or output_base
    combinations = config_module.field_combinations(config)

    if not combinations:
        clustering, node2uuid = run_once(config)
        mapping, _ = results.uuid_to_cluster_map(clustering, node2uuid)
        return results.write_annotations(config, mapping, save_dir)

    mapping, offset = {}, 0
    for field_combo in combinations:
        logger.info(f"Running for field combination: {field_combo}")
        combo_dir = config_module.combination_dir(output_base, field_combo)
        os.makedirs(combo_dir, exist_ok=True)

        clustering, node2uuid = run_once(
            config_module.for_combination(config, field_combo, combo_dir))
        combo_map, offset = results.uuid_to_cluster_map(clustering, node2uuid, offset)
        mapping.update(combo_map)

        # subsequent runs append to the same log
        if isinstance(config.get('logging'), dict):
            config['logging']['file_mode'] = 'a'

    logger.info(f"Combined {len(combinations)} field combinations into {offset} clusters")
    return results.write_annotations(config, mapping, save_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a clustering algorithm named by its config and save the results.")
    parser.add_argument('--config', required=True, help="Path to the config file")
    parser.add_argument('--base_config', default=None,
                        help="Base config (universal); the given config is merged on top")
    parser.add_argument('--save_dir', default=None,
                        help="Where to write the annotation output (default: data.output_path)")
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Print the resolved paths and wait for confirmation before running")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override the config seed; outputs are redirected to a seed<N>/ subdir")
    parser.add_argument('--prob_human_correct', type=float, default=None,
                        help="Override edge_weights.prob_human_correct; outputs are "
                             "redirected to a phc<NN>/ subdir")
    parser.add_argument('--set', dest='overrides', action='append', metavar='KEY=VALUE',
                        help="Override any config key by dotted path, e.g. "
                             "--set stability.target_alpha=0.3 (repeatable)")
    return parser.parse_args(argv)


def main(argv=None):
    init_logger()
    args = parse_args(argv)
    config = config_module.apply_run_tag(
        config_module.load(args.config, args.base_config, args.overrides),
        seed=args.seed, prob_human_correct=args.prob_human_correct)

    if args.interactive:
        data_config = config['data']
        logger.info(f"Config file:      {os.path.abspath(args.config)}")
        logger.info(f"Annotation file:  {data_config.get('annotation_file')}")
        logger.info(f"Output path:      {data_config['output_path']}")
        if data_config.get('separate_by_fields'):
            logger.info(f"Separating by:    {data_config['separate_by_fields']}")
        input("Press Enter to continue...")

    return run_from_config(config, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
