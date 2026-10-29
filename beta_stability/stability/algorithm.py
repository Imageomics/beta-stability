"""The beta-stability algorithm: the loop, and nothing else.

The four steps are those of the paper's "Beta-Stability Algorithm" section
(``sec:phase0andphase1``). Step 1 computes stability for the whole graph, Step 2
changes edges as needed to reach at least 0-stability, Step 3 selects the least
stable pairs for human verification, and Step 4 applies the verdicts. The first
pass through Steps 1 and 2 needs no human input and is Phase 0; the loop over
Steps 3, 4, 1, 2 is Phase 1, named ACTIVE_REVIEW here.

The invariants the implementation maintains:

- only positive edges are ever deactivated, becoming positive-inactive;
- negative edges are never deactivated;
- a review that agrees with the current label adds ch to the edge confidence
  and one that disagrees subtracts it, flipping the label if it goes negative;
- 0-stability is restored after every batch of reviews.

The graph itself, where stability is computed and edges are cut, is
:mod:`beta_stability.stability.graph`. Measuring a run against ground truth is
:mod:`beta_stability.metrics.tracking`, and getting verdicts out of a reviewer is
:mod:`beta_stability.edge_weights.reviewer`. Neither can change what this computes.
"""

from collections import Counter, defaultdict
import networkx as nx
import logging
from typing import Dict, List, Set, Tuple

from beta_stability.stability.graph import StabilityGraph, EdgeLabel
from beta_stability.metrics import tracking
from beta_stability.stability import candidates

logger = logging.getLogger("beta_stability")


class BetaStabilityAlgorithm:
    """
    Beta Stability stability-driven clustering algorithm.

    Implements the standard algorithm interface (step, is_finished, get_clustering).
    """

    def __init__(self, config: Dict, classifier_manager, cluster_validator=None):
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator

        self.graph = StabilityGraph()

        # Run state
        self.phase = "PHASE0"  # PHASE0 -> ACTIVE_REVIEW -> FINISHED
        self.num_human_reviews = 0
        self.human_attempts = {}  # review attempts per edge: {(n0, n1): count}
        self.phase0_iteration = 0
        self._phase0_prev_pcc_count = None
        self._phase0_stall_count = 0
        self._best_alpha_seen = -float('inf')
        self._alpha_stall_batches = 0

        # Parameters. Each one is declared, with its default and what it does,
        # in the stability section of beta_stability.config.DEFAULTS.
        self.target_alpha = config['target_alpha']
        self.phase0_alpha = config['phase0_alpha']
        self.max_phase0_iterations = config['max_phase0_iterations']
        self.cut_margin = config['cut_margin']
        self.max_human_reviews = config['max_human_reviews']
        self.edges_per_review_batch = config['edges_per_review_batch']
        self.ch = config['review_confidence']
        self.cap_confidence = config['cap_confidence']
        self.external_per_batch = config['external_per_batch']
        self.review_weakest_positive = config['review_weakest_positive']
        self.review_positive_inactive = config['review_positive_inactive']
        self.unified_candidate_ranking = config['unified_candidate_ranking']
        self.human_stall_patience = config['human_stall_patience']
        self._stall_epsilon = config['human_stall_epsilon']
        self._stall_min_reviews = config['human_stall_min_reviews']

        # Observing the run is the tracker's business; see beta_stability.metrics.tracking.
        self.tracker = tracking.RunTracker(
            cluster_validator,
            validation_step=config['validation_step'],
            report_at_reviews=config['report_at_reviews'],
            track_stability=config['track_stability'],
            track_path=config['stability_track_path'])

        # Diagnostics. The review tally counts reviews by (error?, verdict,
        # current label, created instability?), which separates the four error
        # kinds: a verdict that flips a negative to positive is a reversible
        # MERGE, one that flips a positive to negative an irreversible SPLIT.
        # Populated in _apply_human_feedback when ground truth exists.
        self._review_tally = Counter()
        self._pending_review_types: Dict[Tuple[int, int], str] = {}
        self._cumulative_agree: Dict[str, int] = defaultdict(int)
        self._cumulative_disagree: Dict[str, int] = defaultdict(int)

        logger.info("Beta Stability Algorithm initialized")
        logger.info(f"  Target alpha: {self.target_alpha}")
        logger.info(f"  Phase 0 alpha: {self.phase0_alpha}")
        logger.info(f"  Cut margin: {self.cut_margin}")
        logger.info(f"  Max Phase 0 iterations: {self.max_phase0_iterations}")
        logger.info(f"  Human confidence (ch): {self.ch}")
        logger.info(f"  Max human reviews: {self.max_human_reviews}")

    def _stabilize(self, alpha: float) -> int:
        """Step 2: deactivate positive edges until the graph is alpha-stable.

        Returns the number of positive edges deactivated.
        """
        return self.graph.make_zero_stable(alpha=alpha)

    def step(self, new_edges: List[Tuple]) -> List[Tuple[int, int, float]]:
        """
        Perform one step of the algorithm.

        Args:
            new_edges: List of (n0, n1, score, verifier_name) tuples

        Returns:
            List of edges needing human review: [(n0, n1, score), ...]
        """
        # Process and add incoming edges
        self._process_and_add_edges(new_edges)

        # Count human reviews and apply feedback
        needs_restabilization = False
        # Track agreement/disagreement per candidate type for informativeness metric
        type_agree = defaultdict(int)
        type_disagree = defaultdict(int)

        for edge in new_edges:
            if len(edge) > 3 and 'human' in str(edge[3]):
                n0, n1, score = edge[:3]

                # Track human review attempts per edge
                edge_key = (min(n0, n1), max(n0, n1))
                self.human_attempts[edge_key] = self.human_attempts.get(edge_key, 0) + 1
                self.num_human_reviews += 1

                # Determine agreement BEFORE applying feedback
                edge_data = self.graph.get_edge(n0, n1)
                if edge_data:
                    human_decision = score > 0.5
                    current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
                    human_agrees = (human_decision == current_is_positive)
                    ctype = self._pending_review_types.get(edge_key, "UNKNOWN")
                    if human_agrees:
                        type_agree[ctype] += 1
                    else:
                        type_disagree[ctype] += 1

                if self._apply_human_feedback(n0, n1, score):
                    needs_restabilization = True
                if self.tracker.checkpoint_due(self.num_human_reviews,
                                               self.graph.G.number_of_nodes()):
                    self._checkpoint(needs_restabilization)

        # Log per-type disagreement rates (higher = more informative candidates)
        all_types = set(type_agree.keys()) | set(type_disagree.keys())
        if all_types:
            # Update cumulative counts
            for ctype in all_types:
                self._cumulative_agree[ctype] += type_agree[ctype]
                self._cumulative_disagree[ctype] += type_disagree[ctype]


            # Batch stats
            parts = []
            for ctype in sorted(all_types):
                a = type_agree[ctype]
                d = type_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (batch): {', '.join(parts)}")

            # Cumulative stats
            all_cumulative = set(self._cumulative_agree.keys()) | set(self._cumulative_disagree.keys())
            parts = []
            for ctype in sorted(all_cumulative):
                a = self._cumulative_agree[ctype]
                d = self._cumulative_disagree[ctype]
                total = a + d
                pct = d / total * 100 if total > 0 else 0
                parts.append(f"{ctype}: {d}/{total} disagree ({pct:.0f}%)")
            logger.info(f"Review informativeness (cumulative): {', '.join(parts)}")

        # Detect PCC splits from human reviews and queue cross-check edges
        # Step 2: update edge labels to make the graph 0-stable again
        if needs_restabilization:
            logger.info("Starting re-stabilization...")
            with tracking.timed('restabilize'):
                deactivations = self._stabilize(alpha=-self.cut_margin)
            if deactivations > 0:
                logger.info(f"Re-stabilized after human reviews: {deactivations} deactivations")

        # Phase dispatch
        if self.phase == "PHASE0":
            return self._phase0_step()
        if self.phase == "ACTIVE_REVIEW":
            return self._active_review_step()
        return []

    def _phase0_step(self) -> List[Tuple[int, int, float]]:
        """One iteration of Phase 0: Steps 1 and 2, without human input.

        Deactivates positive edges until the graph is 0-stable, which is what
        the stability criterion achieves on its own.
        """
        # Increment phase 0 iteration counter
        self.phase0_iteration += 1
        logger.info(f"=== Phase 0 iteration {self.phase0_iteration} ===")

        # Snapshot BEFORE this iteration's stabilization -- iteration 1 captures
        # the raw initial graph (maximal instability), later ones the progressive
        # collapse to 0-stable. This is where most instability is resolved.
        self._snapshot(phase='phase0')

        # Init eval: the raw initial-graph clustering, before any 0-stability
        # change, for the stagewise-decomposition table (logged as "Init").
        if self.phase0_iteration == 1 and self.cluster_validator:
            # graph not needed here -- get_clustering() would copy all 14.5M edges
            _clust, _n2c = self.graph.get_clustering()
            self.cluster_validator.incremental_stats(
                0, _clust, _n2c,
                self.cluster_validator.gt_clustering,
                self.cluster_validator.gt_node2cid, 'Init')


        pccs = self.graph.get_pccs()
        logger.info(f"Found {len(pccs)} PCCs")

        # Make alpha-stable (phase0_alpha controls aggressiveness, default 0.0)
        deactivations = self._stabilize(alpha=self.phase0_alpha - self.cut_margin)
        logger.info(f"Made {deactivations} edges positive-inactive for {self.phase0_alpha}-stability")

        # Log metrics only when significant changes occur (saves ~30s per skipped iteration)
        if deactivations >= 5 or self.phase0_iteration <= 1:
            self.tracker.phase0(*self.graph.get_clustering(), self.phase0_iteration)

        # Check for transition to ACTIVE_REVIEW
        converged = (deactivations == 0)

        # Detect oscillation: PCC count unchanged for consecutive iterations
        current_pcc_count = len(pccs)
        if current_pcc_count == self._phase0_prev_pcc_count:
            self._phase0_stall_count += 1
        else:
            self._phase0_stall_count = 0
        self._phase0_prev_pcc_count = current_pcc_count

        # Force convergence when the iteration limit is reached.
        force_converge = self.phase0_iteration >= self.max_phase0_iterations

        if force_converge and not converged:
            logger.info(f"=== Phase 0 complete - forced (iteration={self.phase0_iteration}, "
                       f"stall_count={self._phase0_stall_count}, deactivations={deactivations}) ===")
            converged = True

        if converged or force_converge:
            logger.info("=== Phase 0 complete - converged ===")
            self._log_graph_stats()

            # Validation after Phase 0 completes
            self._validate()

            if self.target_alpha > 0:
                self.phase = "ACTIVE_REVIEW"
                logger.info("=== Starting Active Review Phase ===")
                return self._active_review_step()

            self.phase = "FINISHED"
            return []

        # Continue Phase 0 (no human review yet)
        return []

    def _compute_adaptive_quotas(self, remaining_budget: int) -> Dict[str, int]:
        """Compute per-bucket slot allocation for the next batch's selectors.

        With experimental selectors stripped, EXTERNAL is the sole non-INTERNAL
        bucket, so it takes the full remaining budget after INTERNAL (capped
        by external_per_batch if set).
        """
        if remaining_budget <= 0:
            return {'EXTERNAL': 0}
        quota = remaining_budget
        if self.external_per_batch is not None:
            quota = min(quota, int(self.external_per_batch))
        return {'EXTERNAL': quota}

    def _active_review_step(self) -> List[Tuple[int, int, float]]:
        """One iteration of Phase 1: Step 3, selecting pairs for verification.

        Computes internal and external stability, orders the candidates by
        increasing stability, and returns the least stable k of them. Step 4,
        applying what the reviewer says, happens back in :meth:`step`.
        """
        # Check termination
        if self.num_human_reviews >= self.max_human_reviews:
            logger.info(f"Reached max human reviews ({self.max_human_reviews})")
            self._finalize('max_human_reviews')
            return []

        # Stability snapshot for this iteration.
        self._snapshot()

        # Get current stability
        with tracking.timed('get_graph_stats'):
            stats = self.graph.get_graph_stats()
        min_internal = stats['min_internal_stability']
        min_external = stats['min_external_stability']

        # Handle inf values correctly:
        # - inf means "perfectly stable" (no conflicts), not "unknown"
        # - Single-element PCCs have inf internal stability (nothing to destabilize)
        # - PCCs with no cross-PCC negative edges have inf external stability
        if min_internal == float('inf') and min_external == float('inf'):
            # Graph is perfectly stable
            current_alpha = float('inf')
        elif min_internal == float('inf'):
            # Internal is perfect, use external only
            current_alpha = min_external
        elif min_external == float('inf'):
            # External is perfect, use internal only
            current_alpha = min_internal
        else:
            current_alpha = min(min_internal, min_external)

        logger.info(f"Current stability: internal={min_internal:.4f}, external={min_external:.4f}")

        if current_alpha >= self.target_alpha:
            logger.info(f"Reached target alpha ({self.target_alpha})")
            self._finalize('target_alpha')
            return []

        # Stall detection: stop early when human reviews stop improving stability
        # (config: human_stall_patience batches; 0 = disabled). This caps the
        # human-review workload of runs that cannot reach target_alpha, without
        # affecting converging runs (which finalize on target_alpha above).
        if self.human_stall_patience > 0:
            if current_alpha > self._best_alpha_seen + self._stall_epsilon:
                self._best_alpha_seen = current_alpha
                self._alpha_stall_batches = 0
            else:
                self._alpha_stall_batches += 1
                if (self._alpha_stall_batches >= self.human_stall_patience
                        and self.num_human_reviews >= self._stall_min_reviews):
                    logger.info(
                        f"No stability gain for {self._alpha_stall_batches} review "
                        f"batches (alpha={current_alpha:.4f}, best="
                        f"{self._best_alpha_seen:.4f}); stopping to bound human review.")
                    self._finalize('stalled_no_progress')
                    return []

        # Generate candidate pools — internal, external — without selecting
        # or capping. Caller (here) owns priority order and budget.
        logger.info("Starting candidate generation...")
        # An edge stays eligible as long as the graph still flags it, and leaves
        # the pool only once review has settled it. A run ends on target_alpha, on
        # running out of candidates, or on the review budget, never on a per-edge
        # quota, since a contested edge may need several reviews to settle.
        with tracking.timed('candidate_pools'):
            pools = candidates.pools(
                self.graph,
                alpha=self.target_alpha,
                include_weak_positive=self.review_weakest_positive,
                review_positive_inactive=self.review_positive_inactive,
            )
        logger.info("Candidate generation complete")

        # Compose the batch in strict priority order under ONE shared budget.
        # Order: INTERNAL → EXTERNAL. Internal owns intra-PCC splits, the
        # cheapest high-value reviews; EXTERNAL fills the remaining slots.
        # Total len(batch) <= edges_per_review_batch, and never more than the
        # reviews left under max_human_reviews, so a reviewer is not shown
        # pairs whose answers the run would no longer use.
        budget = min(self.edges_per_review_batch,
                     self.max_human_reviews - self.num_human_reviews)
        batch: List[Tuple[int, int, float]] = []
        self._pending_review_types = {}
        pccs_used: Set[int] = set()
        used_keys: Set[Tuple[int, int]] = set()
        type_counts: Dict[str, int] = {}
        escalated = 0

        def _add_stability_candidate(candidate) -> bool:
            nonlocal escalated
            """Add a StabilityCandidate respecting per-PCC conflict + dedup.

            INTERNAL/EXTERNAL candidates are 1-per-PCC (the MST cut-point
            is a single decision per cluster).
            """
            if len(batch) >= budget:
                return False
            enforce_pcc_conflict = candidate.pcc_id is not None
            if enforce_pcc_conflict and candidate.pcc_id in pccs_used:
                return False
            u, v = candidate.review_edge
            key = (min(u, v), max(u, v))
            if key in used_keys:
                return False
            edge_data = self.graph.get_edge(u, v)
            score = edge_data.score if edge_data else 0.0

            # A candidate reaches a human only after every automatic verifier
            # downstream of the one that last classified it has had its turn.
            # Escalating to the next verifier costs no seat in the batch. With
            # one automatic verifier configured, every candidate goes straight
            # to review.
            current = edge_data.ranker if edge_data else None
            nxt = self.classifier_manager.get_next_classifier(current)
            if nxt != 'human':
                n0, n1, s, conf, label, ranker = self.classifier_manager.classify_edge(u, v, nxt)
                self.graph.add_edge(
                    n0, n1,
                    EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE,
                    conf, s, ranker)
                escalated += 1
                logger.debug(f"Escalated edge ({u}, {v}) from '{current}' to '{nxt}' "
                             f"instead of human review")
                used_keys.add(key)   # settled this round; do not re-offer it
                return False         # no seat consumed: composition keeps scanning

            batch.append((u, v, score))
            self._pending_review_types[key] = candidate.candidate_type
            used_keys.add(key)
            if enforce_pcc_conflict:
                pccs_used.add(candidate.pcc_id)
            type_counts[candidate.candidate_type] = type_counts.get(candidate.candidate_type, 0) + 1
            return True

        if self.unified_candidate_ranking:
            # Internal and external candidates compete for the same budget by
            # stability, most unstable first. A weak-positive candidate has
            # stability at or above zero, so it takes a seat only once the
            # conflicting pairs are exhausted, rather than displacing merges
            # early in the run.
            ext_cap = self.external_per_batch
            ext_added = 0
            merged = sorted(pools['internal'] + pools['external'],
                            key=lambda c: c.stability)
            for c in merged:
                if len(batch) >= budget:
                    break
                if (c.candidate_type == 'EXTERNAL' and ext_cap is not None
                        and ext_added >= int(ext_cap)):
                    continue
                if _add_stability_candidate(c) and c.candidate_type == 'EXTERNAL':
                    ext_added += 1
        else:
            # Legacy type-priority: INTERNAL claims seats first, EXTERNAL fills
            # the rest (optionally capped). 1-per-PCC enforced.
            for c in pools['internal']:
                if len(batch) >= budget:
                    break
                _add_stability_candidate(c)

            remaining_after_internal = budget - len(batch)
            quotas = self._compute_adaptive_quotas(remaining_after_internal)
            if quotas['EXTERNAL'] > 0:
                ext_quota = min(quotas['EXTERNAL'], budget - len(batch))
                ext_added = 0
                for c in pools['external']:
                    if len(batch) >= budget or ext_added >= ext_quota:
                        break
                    if _add_stability_candidate(c):
                        ext_added += 1

        logger.info(
            f"Batch composition (budget={budget}): "
            f"INTERNAL={type_counts.get('INTERNAL', 0)}, "
            f"WEAK_POSITIVE={type_counts.get('WEAK_POSITIVE', 0)}, "
            f"EXTERNAL={type_counts.get('EXTERNAL', 0)}, "
            f"total={len(batch)}, escalated={escalated}"
        )

        if not batch:
            if escalated:
                # Every candidate still had an automatic verifier left, so the
                # round did algorithmic work and the labels changed. That is
                # progress, not exhaustion: recompute and come back.
                logger.info(f"No edges for review this round: {escalated} candidate(s) "
                            f"escalated to the next automatic verifier")
                return []
            logger.info("No review candidates - graph at maximum stability")
            self._finalize('no_candidates')
            return []

        logger.info(f"Selected {len(batch)} non-conflicting edges for parallel review")

        # Validation
        logger.info("Starting validation...")
        self._validate()
        logger.info("Validation complete")

        logger.info(f"Returning {len(batch)} edges for human review")
        return batch


    def _apply_human_feedback(self, n0: int, n1: int, score: float) -> bool:
        """Apply one verdict to its edge: Step 4.

        A verdict that agrees with the current label adds ch to the confidence,
        one that disagrees subtracts it, and a confidence driven below zero
        flips the label.

        Returns: True if this could create new instability requiring re-stabilization.
        """
        if not self.graph.has_edge(n0, n1):
            return False

        edge_data = self.graph.get_edge(n0, n1)
        human_decision = score > 0.5  # Human says positive if score > 0.5

        # Determine if human agrees with current label
        current_is_positive = edge_data.label in (EdgeLabel.POSITIVE, EdgeLabel.POSITIVE_INACTIVE)
        human_agrees = (human_decision == current_is_positive)

        # Three verdicts can leave the graph unstable and need a re-stabilization:
        # disagreeing with a negative, which flips it to positive and joins two
        # PCCs; agreeing with a positive-inactive, which reactivates it and does
        # the same; and disagreeing with an active positive, which flips it to
        # negative and, if the PCC does not fall apart on that edge alone, leaves
        # a negative inside a PCC.
        was_positive_inactive = edge_data.label == EdgeLabel.POSITIVE_INACTIVE
        was_active_positive = edge_data.label == EdgeLabel.POSITIVE
        could_create_instability = (
            ((not human_agrees) and (not current_is_positive)) or
            ((not human_agrees) and was_active_positive) or
            (human_agrees and was_positive_inactive)
        )

        # Diagnostic tally: categorize this review vs ground truth.
        validator = self.cluster_validator
        gt = getattr(validator, 'gt_node2cid', None) if validator else None
        if gt is not None:
            gu, gv = gt.get(n0), gt.get(n1)
            if gu is not None and gv is not None:
                gt_same = (gu == gv)
                verdict_same = human_decision                 # True = human says "same"
                is_error = (verdict_same != gt_same)
                verdict = 'same' if verdict_same else 'diff'
                label = 'pos' if current_is_positive else 'neg'
                # reversibility: a verdict that flips a positive->negative is a
                # SPLIT (cut-only rectifier cannot undo); negative->positive is a
                # MERGE (rectifier can cut back).
                self._review_tally[(
                    'ERR' if is_error else 'ok', verdict, label,
                    'flip' if could_create_instability else 'noflip')] += 1

        # Apply the review
        self.graph.apply_human_review(n0, n1, human_agrees, self.ch,
                                      cap_confidence=self.cap_confidence)

        logger.info(f"Human review on ({n0},{n1}): decision={'positive' if human_decision else 'negative'}, "
                   f"agrees={human_agrees}")

        return could_create_instability

    def _process_and_add_edges(self, raw_edges: List[Tuple]):
        """Process raw edges and add to graph."""
        for edge in raw_edges:
            if len(edge) < 4:
                continue

            n0, n1, score, verifier_name = edge[:4]

            # Human edges are not added here; _apply_human_feedback applies them.
            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                continue

            classified = self.classifier_manager.classify_edge(n0, n1, verifier_name)
            _, _, score, confidence, label, ranker = classified
            edge_label = EdgeLabel.POSITIVE if label == "positive" else EdgeLabel.NEGATIVE
            self.graph.add_edge(n0, n1, edge_label, confidence, score, ranker)

    def is_finished(self) -> bool:
        """Check if algorithm is finished."""
        if self.phase == "FINISHED":
            return True

        # Phase 0 convergence is handled inside _phase0_step() — don't recompute here
        if self.phase == "PHASE0":
            return False

        # Active review termination
        return self.num_human_reviews >= self.max_human_reviews

    def get_clustering(self) -> Tuple[Dict, Dict, nx.Graph]:
        """Get current clustering results."""
        cluster_dict, node2cid = self.graph.get_clustering()

        # Build NetworkX graph for compatibility
        G = nx.Graph()
        G.add_nodes_from(self.graph.G.nodes())

        for u, v, data in self.graph.G.edges(data=True):
            edge_data = data.get('data')
            if edge_data:
                G.add_edge(u, v,
                          label=edge_data.label.value,
                          confidence=edge_data.confidence,
                          score=edge_data.score,
                          ranker=edge_data.ranker,
                          is_active=(edge_data.label != EdgeLabel.POSITIVE_INACTIVE))

        return cluster_dict, node2cid, G

    # ------------------------------------------------------------------ tracking
    # Three hooks the steps call. Each one gathers the state the tracker needs
    # and hands it over; what gets recorded is decided in beta_stability.metrics.tracking.

    def _validate(self):
        """Score the clustering against ground truth, if enough has changed."""
        if not self.tracker.started:
            self.tracker.start(*self.graph.clustering_with_view(), self.num_human_reviews)
            return
        self.tracker.periodic(*self.graph.clustering_with_view(), self.num_human_reviews)

    def _snapshot(self, phase='active'):
        """Record this iteration's internal and external stabilities."""
        self.tracker.snapshot(
            self.graph.per_pcc_internal_stability(), self.graph.external_stabilities(),
            phase=phase,
            phase0_iteration=self.phase0_iteration if phase == 'phase0' else None,
            reviews=self.num_human_reviews)

    def _checkpoint(self, needs_restabilization):
        """Score the graph at this exact review count, without disturbing the run."""
        scored = self.tracker.checkpoint_on_copy(
            self.graph,
            restabilize=(lambda g: g.make_zero_stable(alpha=-self.cut_margin))
                        if needs_restabilization else None,
            clustering_of=lambda g: g.clustering_with_view()[:2])
        if scored is not None:
            self.tracker.checkpoint(*scored, self.num_human_reviews)

    def _log_graph_stats(self):
        """Log the graph summary, as Phase 0 does on completion."""
        tracking.log_graph_stats(
            self.graph, phase=self.phase, num_human_reviews=self.num_human_reviews,
            target_alpha=self.target_alpha,
            pcc_strength=self.graph.pcc_separation_strength())

    def _finalize(self, reason):
        """Terminate: record why, and have the run scored as it actually ended."""
        self.phase = "FINISHED"
        logger.info(f"TERMINATION: reason={reason} num_human_reviews={self.num_human_reviews}")
        self.tracker.final(*self.graph.clustering_with_view(), self.num_human_reviews)

    def show_stats(self):
        """Log the end-of-run statistics and write the stability track."""
        self._log_graph_stats()
        tracking.log_review_tally(self._review_tally)
        self.tracker.dump()
