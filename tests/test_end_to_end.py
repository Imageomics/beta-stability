"""A real run of the whole pipeline, on a dataset small enough to be fast.

This is the regression net for the algorithm itself. The unit tests pin the
output contract with synthetic inputs, but they feed string-keyed maps the way
JSON would, and the algorithm hands over int-keyed ones -- a difference that
silently assigned no cluster ids until a real run caught it. Anything that
changes the clustering this produces will fail here.

The dataset it runs on is described in conftest.
"""

import pytest

from beta_stability import config as config_module
from beta_stability import run
from tests.conftest import INDIVIDUALS, PER_INDIVIDUAL, read_json

def _config(dataset, output_name):
    root, anno_file, embedding_file = dataset
    output = root / output_name
    return config_module.load_mapping({
        'algorithm_type': 'stability',
        'seed': 0,
        'exp_name': 'e2e',
        'data': {
            'annotation_file': str(anno_file),
            'embedding_file': str(embedding_file),
            'name_keys': ['individual_id'],
            'n_filter_min': 1,
            'n_filter_max': 100,
            'output_path': str(output),
        },
        'edge_weights': {'verifier_names': 'miewid simulated_human',
                         'prob_human_correct': 1.0},
        'logging': {'log_file': str(output / 'run.log'), 'file_mode': 'w'},
        'stability': {'target_alpha': 0.5, 'max_human_reviews': 100,
                      'edges_per_review_batch': 20, 'validation_step': 20},
    })


@pytest.fixture(scope='module')
def completed_run(dataset):
    """Run the pipeline once and return the config and the annotation file."""
    config = _config(dataset, 'out')
    annots_path = run.run_from_config(config)
    return config, annots_path


def test_every_annotation_receives_a_cluster(completed_run):
    _, annots_path = completed_run
    annotations = read_json(annots_path)['annotations']
    assert len(annotations) == INDIVIDUALS * PER_INDIVIDUAL
    unassigned = [a['uuid'] for a in annotations if a['cluster_id'] is None]
    assert not unassigned, f'{len(unassigned)} annotations left without a cluster'


def test_the_input_file_structure_survives(completed_run):
    _, annots_path = completed_run
    produced = read_json(annots_path)
    assert set(produced) == {'categories', 'images', 'annotations'}
    first = produced['annotations'][0]
    assert {'uuid', 'image_uuid', 'viewpoint', 'individual_id', 'species'} <= set(first)


def test_the_graph_files_are_written(completed_run):
    config = completed_run[0]
    output = config['data']['output_path']
    for key in ('clustering_file', 'node2cid_file', 'node2uuid_file', 'graph_file'):
        path = f"{output}/{config['output'][key]}"
        assert read_json(path) is not None, f'{path} is missing'


def test_the_clustering_is_close_to_the_truth(completed_run):
    """Twelve individuals, well separated, so the run should recover about twelve."""
    annots_path = completed_run[1]
    annotations = read_json(annots_path)['annotations']
    clusters = {a['cluster_id'] for a in annotations}
    assert INDIVIDUALS - 2 <= len(clusters) <= INDIVIDUALS + 6, \
        f'recovered {len(clusters)} clusters for {INDIVIDUALS} individuals'

    # no cluster may mix individuals: a perfect reviewer should prevent it
    by_cluster = {}
    for a in annotations:
        by_cluster.setdefault(a['cluster_id'], set()).add(a['individual_id'])
    impure = {c: ids for c, ids in by_cluster.items() if len(ids) > 1}
    assert not impure, f'clusters mixing individuals: {impure}'


def test_the_same_config_gives_the_same_clustering(dataset, completed_run):
    """Determinism, which is what makes this test a regression net at all."""
    first_path = completed_run[1]
    second_path = run.run_from_config(_config(dataset, 'out2'))

    def assignment(path):
        annotations = read_json(path)['annotations']
        return {a['uuid']: a['cluster_id'] for a in annotations}

    assert assignment(second_path) == assignment(first_path)


# --------------------------------------------------------------------------
# without ground truth, which is what a production run looks like
# --------------------------------------------------------------------------

def _config_without_identities(dataset_without_identities, output_name):
    """The same run, but the annotations carry no identity to evaluate against."""
    root, anno_file, embedding_file = dataset_without_identities
    output = root / output_name
    return config_module.load_mapping({
        'algorithm_type': 'stability',
        'seed': 0,
        'exp_name': 'e2e_no_gt',
        'data': {
            'annotation_file': str(anno_file),
            'embedding_file': str(embedding_file),
            'output_path': str(output),
        },
        # no name_keys, so no identities, so no simulated reviewer either
        'edge_weights': {'verifier_names': 'miewid no_human'},
        'logging': {'log_file': str(output / 'run.log'), 'file_mode': 'w'},
        'stability': {'target_alpha': 0.5, 'max_human_reviews': 100,
                      'edges_per_review_batch': 20},
    })


def test_a_run_without_identities_still_clusters(dataset_without_identities):
    """Clustering unlabelled annotations is the production case, not an error."""
    annots_path = run.run_from_config(
        _config_without_identities(dataset_without_identities, 'out'))
    annotations = read_json(annots_path)['annotations']

    assert len(annotations) == INDIVIDUALS * PER_INDIVIDUAL
    unassigned = [a['uuid'] for a in annotations if a['cluster_id'] is None]
    assert not unassigned, f'{len(unassigned)} annotations left without a cluster'
    assert 'individual_id' not in annotations[0], 'the identity really is absent'


def test_asking_for_a_simulated_reviewer_without_identities_says_why(
        dataset_without_identities):
    """It used to fail with a bare pandas KeyError: None."""
    config = _config_without_identities(dataset_without_identities, 'out_sim')
    config['edge_weights']['verifier_names'] = 'miewid simulated_human'

    with pytest.raises(ValueError, match=r'data\.name_keys is unset'):
        run.run_from_config(config)


def test_a_name_keys_typo_names_the_fields_that_do_exist(dataset_without_identities):
    config = _config_without_identities(dataset_without_identities, 'out_typo')
    config['data']['name_keys'] = ['indivdual_id']

    with pytest.raises(KeyError, match='which the annotation file does not have'):
        run.run_from_config(config)


def test_a_reviewer_is_never_asked_past_the_budget_or_twice_after_cant_tell(
        dataset, monkeypatch):
    """Every answer is "can't tell": the run must end within budget, not loop."""
    from beta_stability import algorithm_preparation

    asked = []

    def cant_tell_reviewer(*_args, **_kwargs):
        def review(pairs):
            asked.extend((min(n0, n1), max(n0, n1)) for n0, n1, _ in pairs)
            return [(n0, n1, None) for n0, n1, _ in pairs], False
        return review

    monkeypatch.setattr(algorithm_preparation.reviewer, 'build', cant_tell_reviewer)
    config = _config(dataset, 'cant_tell')
    config['stability']['max_human_reviews'] = 30
    config['stability']['target_alpha'] = 0.99   # the clean data needs no reviews at 0.5
    run.run_from_config(config)
    assert asked, "the run asked for no reviews"
    assert len(asked) <= 30
    assert len(asked) == len(set(asked)), "a can't-tell pair was asked again"
