"""Loading, merging and defaulting of run configs.

Every default the pipeline relies on is declared once, in :data:`DEFAULTS`,
rather than written at each use site. A run config is that dict with the base
config merged over it, then the species config, then any command-line
overrides, so a key's value is always the most specific one provided.

Merging, interpolation and dotted-path overrides come from OmegaConf. The
result is converted back to a plain dict before the pipeline sees it, because
the rest of the package expects ordinary dicts.

Interpolation is available in any config, for example::

    data:
      output_path: results/${algorithm_type}
"""

import itertools
import logging
import os

from omegaconf import OmegaConf

from beta_stability.data.io import (
    discover_field_values_from_df,
    load_dataframe_lightweight,
)

logger = logging.getLogger("beta_stability")

#: Every default the runner depends on. A config may override any of these.
DEFAULTS = {
    'algorithm_type': 'stability',
    'seed': 42,
    'run': {
        # Set from the command line; names the subdirectory a sweep writes into.
        'tag': '',
    },
    'data': {
        'output_path': 'tmp',
        'id_key': 'uuid',
        # The annotation fields that form the ground-truth identity. Null means
        # the annotations carry none, so the run clusters without evaluating
        # itself, which is the production case.
        'name_keys': None,
        # Drop individuals with fewer annotations than this, and subsample
        # those with more. Both need identities, so both are ignored without
        # name_keys. Null means no filtering.
        'n_filter_min': None,
        'n_filter_max': None,
        # Where the images live, for the runs that need to look at them.
        'images_dir': None,
        'format': 'drone',
        'output_key': 'cluster_id',
    },
    'algorithm': {
        'max_outer_iterations': 100,
        # The initial edge set: the top k most similar pairs per node, and the
        # bottom k least similar. 'auto' derives k from the coupon-collector
        # model below.
        'initial_topk': 10,
        'initial_botk': 0,
        'auto_topk_coverage': 0.9,
        'target_edges': 0,
    },
    # The beta-stability algorithm's own parameters. Every one of these was
    # previously defaulted at its use site, several of them twice and with
    # different values, so a reader could not tell what a run would use.
    'stability': {
        # Read from this section because it is the algorithm-type's own.
        'verifier_name': 'miewid',
        'auto_verifier_primary': 'miewid',
        # Below this silhouette at K-hat the embedding is not cleanly
        # clusterable, and the run switches to a k-means verifier.
        'auto_verifier_silhouette_threshold': 0.15,
        # Answer reviews from the identity labels rather than asking a person.
        'simulate_human': True,
        'distance_power': 1,
        # The stability target. A PCC is accepted once every internal and
        # external decision is stable by this margin.
        'target_alpha': 0.5,
        # Phase 0 runs to a weaker target, on classifier evidence alone.
        'phase0_alpha': 0.0,
        # 'auto' derives the cap from the data; every current experiment uses it.
        'max_phase0_iterations': 'auto',
        # Deactivate a positive edge only when the crossing negative beats the
        # weakest positive on its MSP by this margin; marginal negatives go to
        # review instead of being cut on classifier evidence alone.
        'cut_margin': 0.0,
        # The review budget, and how many edges go out in one batch.
        'max_human_reviews': 1000,
        'edges_per_review_batch': 200,
        # Confidence a review contributes, 'auto' to derive it from the
        # candidate count (see auto_compute_stability_params).
        'review_confidence': 0.97,
        'cap_confidence': True,
        # Cap on external candidates per batch; null absorbs whatever is left
        # after the internal ones.
        'external_per_batch': None,
        # The two secondary selectors, both part of the published baseline.
        'review_weakest_positive': True,
        'review_positive_inactive': True,
        # Fill the batch from internal and external candidates ranked together
        # by stability, rather than giving internal ones priority.
        'unified_candidate_ranking': True,
        # Terminate early when min stability has not improved by
        # human_stall_epsilon for this many consecutive batches. 0 = disabled.
        'human_stall_patience': 0,
        'human_stall_epsilon': 1e-4,
        'human_stall_min_reviews': 0,
        # Validation against ground truth: every N reviews, and at these exact
        # review counts ('N' = one review per annotation).
        'validation_step': 100,
        'report_at_reviews': ['N'],
        # The raw per-iteration stability track, for offline plots.
        'track_stability': False,
        'stability_track_path': None,
        # Load the algorithm from internal/stability_algorithm_ablation.py.
        'use_ablation_module': False,
        # Smallest cluster k-means is allowed to imply, which bounds the K it
        # may be asked for.
        'kmeans_min_cluster_size': 1,
        'kmeans_fallback_cluster_size': 1,
    },
    'edge_weights': {
        # Probability the simulated reviewer answers correctly.
        'prob_human_correct': 0.98,
        # Which kinds of mistake a wrong verdict may be: 'merge_only',
        # 'split_only' or either.
        'error_mode': 'both',
        'scorer': 'kde',
        # Percentile used as the threshold when the GMM fit is rejected.
        'fallback_percentile': 99,
        'augmentation_names': 'miewid human',
        # UI reviewer: hold the answers until every pair of the current batch is
        # answered, then re-stabilize once, instead of after each answer.
        'ui_wait_for_full_batch': False,
    },
    'logging': {
        'log_level': 'INFO',
        'file_mode': 'w',
        'update_log_file': True,
    },
    # The baselines. Each is selected by algorithm_type and reads the section
    # named after it, so its parameters are declared here like any other.
    'manual_review': {
        'verifier_name': 'miewid',
        'distance_power': 1,
        'simulate_human': True,
        'topk': 10,
        # None reviews every candidate, which is what makes this manual review.
        'high_threshold': None,
        'max_reviews': None,
        'review_batch_size': 100,
        'clustering_method': 'connected_components',
        # Evaluate every N reviews, which by default is once per batch.
        'eval_every_reviews': '${manual_review.review_batch_size}',
        'report_at_reviews': ['N'],
        # Skip a pair once a confirmed path already joins its two annotations.
        'stop_at_match': True,
    },
    'thresholded_review': {
        'verifier_name': 'miewid',
        'distance_power': 1,
        'simulate_human': True,
        'topk': 10,
        # Accept a pair above this without review; below it, ask.
        'high_threshold': 0.9,
        'low_threshold': 0.65,
        'max_reviews': None,
        'review_batch_size': 100,
        'clustering_method': 'connected_components',
        # Evaluate every N reviews, which by default is once per batch.
        'eval_every_reviews': '${thresholded_review.review_batch_size}',
        'report_at_reviews': ['N'],
        'stop_at_match': True,
    },
    'hdbscan': {
        'verifier_name': 'miewid',
        'distance_power': 1,
        'simulate_human': True,
        'min_cluster_size': 2,
        'min_samples': None,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'cluster_selection_epsilon': 0.0,
        'allow_single_cluster': False,
    },
    'nis': {
        'verifier_name': 'miewid',
        'distance_power': 1,
        'simulate_human': True,
        'n_sampled_vertices': 50,
        'm_neighbors_per_vertex': 100,
        # Neighbours per vertex as a multiple of sampled vertices, when the
        # two above are derived rather than given.
        'nis_nv_nn_ratio': 7,
        'max_human_reviews': 5000,
        'edges_per_review_batch': 200,
        'validation_step': 100,
    },
    'np3_aas': {
        'verifier_name': 'miewid',
        'distance_power': 1,
        'simulate_human': True,
        'dbscan_eps': 0.5,
        'dbscan_min_samples': 2,
        'dbscan_metric': 'cosine',
        's_min': 0.3,
        'k_max': 5,
        'epsilon': 0.6,
        'max_human_reviews': 5000,
        'edges_per_review_batch': 200,
        'validation_step': 100,
    },
    'output': {
        'prefix': None,
        'suffix': None,
        # The per-run graph files, read by the analysis scripts.
        'clustering_file': 'clustering.json',
        'node2cid_file': 'node2cid.json',
        'node2uuid_file': 'node2uuid_file.json',
        'graph_file': 'graph.json',
        # The annotation file a run produces when no prefix or suffix is set.
        'annotations_name': 'beta_stability_annots.json',
    },
}

#: Output paths that a tagged run redirects into its own subdirectory.
TAGGED_PATHS = (
    ('data', 'output_path'),
    ('logging', 'log_file'),
    ('logging', 'auto_threshold_plot_path'),
    ('stability', 'stability_track_path'),
)


def _resolve(*layers):
    """Merge config layers, most specific last, into a plain resolved dict."""
    return OmegaConf.to_container(OmegaConf.merge(*layers), resolve=True)


def load_mapping(mapping):
    """Build a run config from an in-memory mapping, over the declared defaults."""
    return _resolve(OmegaConf.create(DEFAULTS), OmegaConf.create(mapping))


def load(config_path, base_path=None, overrides=None):
    """Build a run config: defaults, then base, then config, then overrides.

    Args:
        config_path: the species or experiment config.
        base_path: optional universal config merged under ``config_path``.
        overrides: optional dotted-path assignments, e.g. ``['seed=7']``.

    Returns:
        dict: a plain, fully resolved config.
    """
    layers = [OmegaConf.create(DEFAULTS)]
    if base_path:
        layers.append(OmegaConf.load(base_path))
    layers.append(OmegaConf.load(config_path))
    if overrides:
        layers.append(OmegaConf.from_dotlist(list(overrides)))
    return _resolve(*layers)


def run_tag(seed=None, prob_human_correct=None):
    """The subdirectory name for one point of a sweep, e.g. ``phc95/seed7``."""
    parts = []
    if prob_human_correct is not None:
        parts.append(f'phc{round(prob_human_correct * 100)}')
    if seed is not None:
        parts.append(f'seed{seed}')
    return os.path.join(*parts) if parts else ''


def apply_run_tag(config, seed=None, prob_human_correct=None):
    """Record a sweep point in the config and send its outputs to their own place.

    One config serves many seeds and reviewer accuracies, so each combination
    writes into ``phc<NN>/seed<N>/`` instead of overwriting the previous one.
    """
    if prob_human_correct is not None:
        config.setdefault('edge_weights', {})['prob_human_correct'] = prob_human_correct
    if seed is not None:
        config['seed'] = seed

    tag = run_tag(seed, prob_human_correct)
    config.setdefault('run', {})['tag'] = tag
    if not tag:
        return config

    for section, key in TAGGED_PATHS:
        values = config.get(section)
        if not (isinstance(values, dict) and values.get(key)):
            continue
        path = os.path.join(os.path.dirname(values[key]), tag, os.path.basename(values[key]))
        values[key] = path
        directory = path if key == 'output_path' else os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    return config


def field_combinations(config):
    """The field-value combinations a config asks to be run separately.

    A config either lists the values for a field as ``<field>_list`` or leaves
    them to be discovered from the annotations. Returns an empty list when the
    config asks for a single combined run.
    """
    data_params = config['data']
    fields = data_params.get('separate_by_fields')
    if fields is None and data_params.get('separate_viewpoints', False):
        fields = ['viewpoint']
    if not fields:
        return []

    values = {f: data_params[f"{f}_list"] for f in fields if f"{f}_list" in data_params}
    undiscovered = [f for f in fields if f not in values]
    if undiscovered:
        discovered = discover_field_values_from_df(
            load_dataframe_lightweight(config), undiscovered)
        values.update(discovered)
        for field, found in discovered.items():
            logger.info(f"Discovered {len(found)} values for '{field}': {found}")

    names = list(values)
    return [dict(zip(names, combo, strict=True))
            for combo in itertools.product(*(values[n] for n in names))]


def for_combination(config, field_combo, output_path):
    """A copy of the config restricted to one field combination."""
    sub = dict(config)
    sub['data'] = dict(config['data'])
    for field, value in field_combo.items():
        sub['data'][f"{field}_list"] = value if isinstance(value, list) else [value]
    sub['data']['output_path'] = output_path
    if 'histogram_path' in sub.get('logging', {}):
        sub['logging'] = dict(sub['logging'])
        sub['logging']['histogram_path'] += f"_{field_combo}.png"
    return sub


def combination_dir(output_base, field_combo):
    """Where one field combination writes its graph files."""
    name = '_'.join(f"{k}-{v}" for k, v in field_combo.items())
    return os.path.join(output_base, name.replace(':', '_').replace('/', '_'))
