"""The UI reviewer never shows more pairs than the review budget allows.

A person answering pairs in the UI should only see pairs whose answers the
run will use: the batch is cut to the remaining budget, and whatever is left
unanswered when the run ends is taken back out of the database.
"""

import sqlite3

import pandas as pd
import pytest

from beta_stability.edge_weights.human_db import human_db


@pytest.fixture
def reviewer_factory(tmp_path):
    uuids = [f"u{i}" for i in range(6)]
    df = pd.DataFrame({'uuid': uuids, 'image_path': [f"{u}.jpg" for u in uuids]})
    node2uuid = dict(enumerate(uuids))
    db_path = tmp_path / "reviews.db"

    def make(max_reviews):
        return human_db(str(db_path), df, node2uuid, max_reviews=max_reviews)
    return make, db_path


def _rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT id, status, decision FROM image_verification").fetchall()


def _answer_all(db_path, decision='correct'):
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE image_verification SET status='checked', decision=?", (decision,))


def test_a_batch_is_cut_to_the_budget(reviewer_factory):
    make, db_path = reviewer_factory
    make(max_reviews=2)([(0, 1, 0.9), (0, 2, 0.8), (0, 3, 0.7)])
    assert len(_rows(db_path)) == 2


def test_waiting_and_answered_pairs_both_hold_a_slot(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=3)
    review([(0, 1, 0.9), (0, 2, 0.8)])
    _answer_all(db_path)
    verdicts, _ = review([(0, 3, 0.7), (0, 4, 0.6), (0, 5, 0.5)])
    assert len(verdicts) == 2, 'the two answers are still read back'
    assert len(_rows(db_path)) == 3, 'only one new pair fits in the budget'


def test_pairs_already_in_the_database_keep_being_read(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=1)
    review([(0, 1, 0.9)])
    _answer_all(db_path)
    review([(0, 1, 0.9)])            # consumed: checked -> sent
    verdicts, _ = review([(0, 1, 0.9)])
    assert verdicts == [(0, 1, True)]


def test_no_budget_means_no_cut(reviewer_factory):
    make, db_path = reviewer_factory
    make(max_reviews=None)([(0, 1, 0.9), (0, 2, 0.8), (0, 3, 0.7)])
    assert len(_rows(db_path)) == 3


def test_unanswered_pairs_are_withdrawn_at_the_end(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=None)
    review([(0, 1, 0.9), (0, 2, 0.8)])
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE image_verification SET status='checked', decision='incorrect' "
                     "WHERE id='u0____u1'")
    assert review.withdraw_pending() == 1
    assert _rows(db_path) == [('u0____u1', 'checked', 'incorrect')]


def test_cant_tell_carries_no_verdict(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=None)
    review([(0, 1, 0.9), (0, 2, 0.8), (0, 3, 0.7)])
    with sqlite3.connect(db_path) as conn:
        conn.executemany("UPDATE image_verification SET status='checked', decision=? WHERE id=?",
                         [('correct', 'u0____u1'), ('incorrect', 'u0____u2'),
                          ('cant_tell', 'u0____u3')])
    verdicts, _ = review([])
    assert sorted(verdicts) == [(0, 1, True), (0, 2, False), (0, 3, None)]


def test_answers_from_an_earlier_run_hold_no_slot_until_read(reviewer_factory):
    make, db_path = reviewer_factory
    first = make(max_reviews=2)
    first([(0, 1, 0.9), (0, 2, 0.8)])
    _answer_all(db_path)
    first([])                             # both answers read: checked -> sent
    restarted = make(max_reviews=2)
    restarted([(0, 3, 0.7), (0, 4, 0.6)])
    assert len(_rows(db_path)) == 4, 'the old sent answers must not block new pairs'


def _priorities(db_path):
    with sqlite3.connect(db_path) as conn:
        return dict(conn.execute("SELECT id, priority FROM image_verification "
                                 "WHERE status = 'awaiting'").fetchall())


def test_waiting_pairs_are_ranked_in_the_order_the_algorithm_asked(reviewer_factory):
    make, db_path = reviewer_factory
    make(max_reviews=None)([(0, 3, 0.7), (0, 1, 0.9), (0, 2, 0.8)])
    assert _priorities(db_path) == {'u0____u3': 0, 'u0____u1': 1, 'u0____u2': 2}


def test_pairs_the_next_batch_drops_are_withdrawn(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=None)
    review([(0, 1, 0.9), (0, 2, 0.8)])
    review([(0, 3, 0.7), (0, 2, 0.8)])
    assert _priorities(db_path) == {'u0____u3': 0, 'u0____u2': 1}


def test_only_this_runs_pairs_are_withdrawn(reviewer_factory):
    make, db_path = reviewer_factory
    other = make(max_reviews=None)
    other([(4, 5, 0.5)])                    # e.g. a postprocessing pair in a shared table
    review = make(max_reviews=None)
    review([(0, 1, 0.9)])
    review([(0, 2, 0.8)])
    review.withdraw_pending()
    assert set(_priorities(db_path)) == {'u4____u5'}


def test_outstanding_counts_this_runs_unanswered_pairs(reviewer_factory):
    make, db_path = reviewer_factory
    review = make(max_reviews=None)
    review([(0, 1, 0.9), (0, 2, 0.8)])
    assert review.outstanding() == 2
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE image_verification SET status='checked', decision='correct' "
                     "WHERE id='u0____u1'")
    assert review.outstanding() == 1
