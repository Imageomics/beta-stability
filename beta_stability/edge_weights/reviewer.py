"""Getting human answers for annotation pairs.

A reviewer is a callable: given a batch of pairs, it returns a verdict per
pair and whether it wants to stop. Three kinds exist, selected by the config:

* **simulated** — reads the identity labels and answers correctly with
  probability ``prob_human_correct``, which is how every experiment in the
  paper is run;
* **interactive** — serves the pairs through the review UI's database and
  waits for a person;
* **none** — answers nothing, for the fully automatic runs.

:func:`build` picks between them. :func:`responses` converts the verdicts back
into the scored edges the algorithm's loop consumes.
"""

import logging
import random

from beta_stability.edge_weights.human_db import human_db

logger = logging.getLogger("beta_stability")


# --------------------------------------------------------------------------
# the simulated reviewer
# --------------------------------------------------------------------------

def get_review(node_1, node_2, df, name_key, rate=0.98, error_mode='both'):
    """One simulated verdict: are these two annotations the same individual?

    Correct with probability ``rate``. When the answer is wrong, ``error_mode``
    restricts which kind of error may fire:

    * ``'merge_only'`` — only false merges, so a truly different pair is called
      the same, and false splits on truly-same pairs are suppressed;
    * ``'split_only'`` — only false splits;
    * ``'both'`` — either.
    """
    is_similar = df.iloc[node_1][name_key] == df.iloc[node_2][name_key]

    if random.random() < rate:
        return is_similar
    if error_mode == 'merge_only' and is_similar:
        return is_similar       # suppress the irreversible false split
    if error_mode == 'split_only' and not is_similar:
        return is_similar       # suppress the reversible false merge
    return not is_similar


def simulated(df, name_key, rate, error_mode='both'):
    """A reviewer that answers from the identity labels, erring at ``1 - rate``."""
    def review(edge_nodes):
        return [(n0, n1, get_review(n0, n1, df, name_key, rate, error_mode))
                for n0, n1, _ in edge_nodes], False
    return review


def silent(quit_run):
    """A reviewer that answers nothing.

    ``quit_run`` says whether the run should stop: a config that asks for
    ``no_human`` wants the algorithm to continue automatically, whereas a
    config that names no reviewer at all has nothing further to contribute.
    """
    def review(_edge_nodes):
        return [], quit_run
    return review


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def build(verifier_names, *, df, filtered_df, node2uuid, name_key,
          prob_human_correct, error_mode='both', ui_db_path=None, simulate_human=True,
          max_reviews=None):
    """Build the reviewer the config asks for.

    ``verifier_names`` is the ordered verifier list; the first human entry in
    it decides the kind. ``'human'`` is the older spelling, kept because the
    species configs still use it, and it means simulated unless
    ``simulate_human`` is false. ``max_reviews`` caps how many pairs the UI
    reviewer puts in front of a person.
    """
    def simulated_reviewer():
        if not name_key:
            raise ValueError(
                "A simulated reviewer answers from the ground-truth identities, "
                "but data.name_keys is unset so there are none. Either set "
                "data.name_keys, or ask for 'no_human' or 'ui_human' in "
                "edge_weights.verifier_names.")
        return simulated(df, name_key, prob_human_correct, error_mode)

    def interactive_reviewer():
        logger.info(f"ui_human - using UI database for human reviews at {ui_db_path}")
        return human_db(ui_db_path, filtered_df, node2uuid, max_reviews=max_reviews)

    for name in verifier_names:
        if name == 'simulated_human':
            return simulated_reviewer()
        if name == 'ui_human':
            if ui_db_path:
                return interactive_reviewer()
            logger.warning("ui_human specified but no ui_db_path provided, "
                           "falling back to simulated")
            return simulated_reviewer()
        if name == 'no_human':
            logger.info("no_human - running without human reviews")
            return silent(quit_run=False)

    if 'human' in verifier_names:
        if simulate_human:
            return simulated_reviewer()
        if ui_db_path:
            return interactive_reviewer()
        return simulated_reviewer()

    logger.info("No human reviewer type specified, defaulting to no_human")
    return silent(quit_run=True)


# --------------------------------------------------------------------------
# the loop's interface to whichever reviewer was built
# --------------------------------------------------------------------------

def responses(requested_edges, reviewer):
    """Put the requested pairs to the reviewer and score the verdicts.

    The reviewer answers same or different; the algorithm's loop consumes
    edges. This is the only place the two formats meet.
    """
    if requested_edges and len(requested_edges[0]) == 4:
        requested_edges = [(e[0], e[1], e[2]) for e in requested_edges]

    verdicts, quit_requested = reviewer(requested_edges)
    scored = [(n0, n1, 1.0 if verdict else 0.0, 'human')
              for n0, n1, verdict in verdicts]
    return scored, quit_requested
