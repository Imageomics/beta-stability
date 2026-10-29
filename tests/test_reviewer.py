"""Which reviewer a config gets, and what the loop receives back.

The selection has six outcomes and the configs in use exercise several of
them, including the older ``human`` spelling. Getting it wrong means a run
either silently stops or silently reviews nothing, so each branch is pinned.
"""

import pandas as pd
import pytest

from beta_stability.edge_weights import reviewer


@pytest.fixture
def labels():
    """Four annotations: a and b are one individual, c and d are two others."""
    return pd.DataFrame({'name': ['x', 'x', 'y', 'z']})


def _build(names, labels, **kwargs):
    defaults = {'df': labels, 'filtered_df': labels, 'node2uuid': {},
                'name_key': 'name', 'prob_human_correct': 1.0}
    return reviewer.build(names, **{**defaults, **kwargs})


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def test_simulated_human_reads_the_labels(labels):
    review = _build(['miewid', 'simulated_human'], labels)
    verdicts, quit_requested = review([(0, 1, 0.9), (0, 2, 0.4)])
    assert [v for _, _, v in verdicts] == [True, False]
    assert quit_requested is False


def test_no_human_answers_nothing_but_lets_the_run_continue(labels):
    verdicts, quit_requested = _build(['miewid', 'no_human'], labels)([(0, 1, 0.9)])
    assert verdicts == []
    assert quit_requested is False, 'the automatic run must not be stopped'


def test_naming_no_reviewer_stops_the_run(labels):
    verdicts, quit_requested = _build(['miewid'], labels)([(0, 1, 0.9)])
    assert verdicts == []
    assert quit_requested is True, 'nothing further to contribute, so stop'


def test_legacy_human_means_simulated(labels):
    review = _build(['miewid', 'human'], labels)
    verdicts, _ = review([(0, 1, 0.9)])
    assert [v for _, _, v in verdicts] == [True]


def test_legacy_human_without_simulation_falls_back_when_no_db(labels):
    review = _build(['human'], labels, simulate_human=False, ui_db_path=None)
    verdicts, _ = review([(2, 3, 0.9)])
    assert [v for _, _, v in verdicts] == [False], 'simulated, not silent'


def test_ui_human_without_a_database_falls_back_to_simulated(labels):
    review = _build(['ui_human'], labels, ui_db_path=None)
    verdicts, _ = review([(0, 1, 0.9)])
    assert [v for _, _, v in verdicts] == [True]


def test_the_first_human_entry_wins(labels):
    """no_human before simulated_human means no reviews, and the reverse too."""
    assert _build(['no_human', 'simulated_human'], labels)([(0, 1, 0.5)])[0] == []
    assert _build(['simulated_human', 'no_human'], labels)([(0, 1, 0.5)])[0] != []


# --------------------------------------------------------------------------
# the error model
# --------------------------------------------------------------------------

def test_a_perfect_reviewer_never_errs(labels):
    review = reviewer.simulated(labels, 'name', rate=1.0)
    pairs = [(0, 1, 0.0), (0, 2, 0.0), (2, 3, 0.0)]
    assert [v for _, _, v in review(pairs)[0]] == [True, False, False]


def test_an_always_wrong_reviewer_inverts_every_verdict(labels):
    review = reviewer.simulated(labels, 'name', rate=0.0)
    assert [v for _, _, v in review([(0, 1, 0.0), (0, 2, 0.0)])[0]] == [False, True]


def test_merge_only_errors_never_split_a_true_match(labels):
    review = reviewer.simulated(labels, 'name', rate=0.0, error_mode='merge_only')
    verdicts = {(n0, n1): v for n0, n1, v in review([(0, 1, 0.0), (0, 2, 0.0)])[0]}
    assert verdicts[(0, 1)] is True, 'a true match is never called different'
    assert verdicts[(0, 2)] is True, 'a true non-match may be called the same'


def test_split_only_errors_never_merge_a_true_non_match(labels):
    review = reviewer.simulated(labels, 'name', rate=0.0, error_mode='split_only')
    verdicts = {(n0, n1): v for n0, n1, v in review([(0, 1, 0.0), (0, 2, 0.0)])[0]}
    assert verdicts[(0, 1)] is False, 'a true match may be called different'
    assert verdicts[(0, 2)] is False, 'a true non-match is never called the same'


# --------------------------------------------------------------------------
# the loop's interface
# --------------------------------------------------------------------------

def test_responses_scores_verdicts_and_tags_them_human():
    def review(pairs):
        return [(n0, n1, n0 == 0) for n0, n1, _ in pairs], False

    scored, quit_requested = reviewer.responses([(0, 1, 0.7), (2, 3, 0.2)], review)
    assert scored == [(0, 1, 1.0, 'human'), (2, 3, 0.0, 'human')]
    assert quit_requested is False


def test_responses_accepts_four_tuples_from_the_algorithm():
    seen = []

    def review(pairs):
        seen.extend(pairs)
        return [], False

    reviewer.responses([(0, 1, 0.7, 'stability')], review)
    assert seen == [(0, 1, 0.7)], 'the source tag is not the reviewer''s business'


def test_responses_passes_the_quit_request_through():
    assert reviewer.responses([(0, 1, 0.5)], lambda _pairs: ([], True))[1] is True


def test_responses_keeps_cant_tell_as_a_none_score():
    scored, _ = reviewer.responses([(0, 1, 0.7)], lambda _pairs: ([(0, 1, None)], False))
    assert scored == [(0, 1, None, 'human')], "can't tell must not read as different"
