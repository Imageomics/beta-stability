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
