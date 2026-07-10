"""
Rewirer module.

Provides clique-based rewiring of undirected graphs while preserving
the original mean degree within a configurable tolerance band.
"""
from __future__ import annotations

import random as rng
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import networkx as nx


OUTCOME_ACCEPTED = "accepted"
OUTCOME_FULL_CLIQUE = "full_clique_groups"
OUTCOME_TOLERANCE_REJECTED = "tolerance_rejections"


def _init_loop_state(config: Dict[str, object]):
    """
    Build the mutable state dictionary used throughout the rewiring loop.

    @param config: Run configuration from _setup_rewiring_state.
    @return: Mutable loop state initialised to starting values.
    """
    return {
        "edge_count": config["edge_count"],
        "final_avg_degree": config["original_avg_degree"],
        "consecutive_failures": 0,
        "total_attempts": 0,
        "cached_group_hits": 0,
        "full_clique_groups": 0,
        "tolerance_rejections": 0,
        "successful_rewirings": 0,
        "checked_groups": set(),
    }


def _build_run_stats(
    config: Dict[str, object],
    state: Dict[str, object],
):
    """
    Assemble the public-facing statistics dictionary from run data.

    @param config: Immutable run configuration.
    @param state: Final loop state from the rewiring loop.
    @return: Flat dict of labelled statistics.
    """
    return {
        "node_count": config["node_count"],
        "target_clique_size": config["target_clique_size"],
        "allowed_clique_sizes": config["allowed_clique_sizes"],
        "degree_tolerance": config["degree_tolerance"],
        "original_avg_degree": config["original_avg_degree"],
        "final_avg_degree": state["final_avg_degree"],
        "total_attempts": state["total_attempts"],
        "successful_rewirings": state["successful_rewirings"],
        "cache_hits": state["cached_group_hits"],
        "already_clique_groups": state["full_clique_groups"],
        "tolerance_rejections": state["tolerance_rejections"],
        "termination_reason": state["termination_reason"],
    }


def _tolerance_exceeded(
    new_avg_degree: float,
    original_avg_degree: float,
    degree_tolerance: float,
):
    """
    Return True if the proposed mean degree violates the tolerance band.

    @param new_avg_degree: Projected mean degree after candidate insertion.
    @param original_avg_degree: Mean degree of the unmodified graph.
    @param degree_tolerance: Allowed relative deviation (0.1 == 10%).
    @return: True if tolerance is exceeded.
    """
    if original_avg_degree == 0:
        return new_avg_degree != 0
    return (
        abs(new_avg_degree - original_avg_degree) / original_avg_degree
        > degree_tolerance
    )


def _loop_should_continue(state: Dict[str, object], config: Dict[str, object]):
    """
    Return True while the rewiring loop may still make progress.

    @param state: Current mutable loop state.
    @param config: Immutable run configuration.
    @return: True if neither stopping criterion has been reached.
    """
    return bool(
        state["consecutive_failures"] < config["max_consecutive_failures"]
        and state["total_attempts"] < config["max_total_attempts"]
    )


def _get_termination_reason(
    state: Dict[str, object],
    config: Dict[str, object],
):
    """
    Determine which stopping criterion ended the rewiring loop.

    @param state: Final mutable loop state.
    @param config: Immutable run configuration.
    @return: 'max_consecutive_failures' or 'max_total_attempts'.
    """
    if state["consecutive_failures"] >= config["max_consecutive_failures"]:
        return "max_consecutive_failures"
    return "max_total_attempts"


class Rewirer:
    """
    Rewires an undirected graph by inserting cliques of a target size.

    Only insertions that keep the mean degree within *degree_tolerance* of
    the original value are accepted.  Termination is guaranteed by two
    independent counters: consecutive failures and a global attempt cap.

    @param ntwk: The undirected graph to rewire (modified in-place).
    @type  ntwk: nx.Graph
    """

    def __init__(self, ntwk: nx.Graph):
        """
        Initialise the rewirer with a target graph.

        @param ntwk: The undirected graph to rewire.
        @type  ntwk: nx.Graph
        """
        self.ntwk = ntwk
        self.last_run_stats = None

    def rewire_network_cliques(
        self,
        target_clique_size: int = 5,
        degree_tolerance: float = 0.1,
        print_stats: bool = False,
    ):
        """
        Insert cliques of *target_clique_size* nodes into the graph.

        Accepts a candidate group only if adding its missing edges keeps
        the mean degree within *degree_tolerance*.  Uses a group cache to
        avoid re-evaluating the same candidate in the same graph state.

        @param target_clique_size: Nodes per clique (min 2, max node_count).
        @param degree_tolerance: Allowed relative deviation (0.1 == ±10%).
        @param print_stats: Print a performance summary after completion.
        @return: The rewired graph (same object as self.ntwk).
        @raises ValueError: On disconnected graph or invalid clique size.
        """
        config = self._setup_rewiring_state(target_clique_size, degree_tolerance)
        loop_result = self._run_rewiring_loop(config)
        self.last_run_stats = _build_run_stats(config, loop_result)
        if print_stats:
            self._print_run_stats()
        return self.ntwk

    def get_last_run_stats(self):
        """
        Return the statistics dictionary from the most recent run.

        @return: Statistics dict, or None if no run has been performed.
        @rtype:  dict or None
        """
        return self.last_run_stats

    def _setup_rewiring_state(
        self,
        target_clique_size: int,
        degree_tolerance: float,
    ):
        """
        Validate preconditions and build the immutable run configuration.

        @param target_clique_size: Requested clique size.
        @param degree_tolerance: Allowed relative degree deviation.
        @return: Configuration dict consumed by the rewiring loop.
        @rtype:  dict
        """
        self._validate_preconditions(target_clique_size)
        nodes = list(self.ntwk.nodes())
        node_count = len(nodes)
        rng.shuffle(nodes)
        max_consecutive_failures = node_count * 10
        allowed_clique_sizes = self._build_allowed_clique_sizes(
            target_clique_size,
            node_count,
        )
        return {
            "nodes": nodes,
            "node_count": node_count,
            "target_clique_size": target_clique_size,
            "allowed_clique_sizes": allowed_clique_sizes,
            "degree_tolerance": degree_tolerance,
            "original_avg_degree": self._calculate_average_degree(),
            "edge_count": self.ntwk.number_of_edges(),
            "max_consecutive_failures": max_consecutive_failures,
            "max_total_attempts": max(
                node_count * max_consecutive_failures, max_consecutive_failures
            ),
        }

    def _run_rewiring_loop(self, config: Dict[str, object]):
        """
        Drive the main rewiring loop until no further progress can be made.

        @param config: Immutable run configuration.
        @return: Final loop state containing all counters and termination reason.
        @rtype:  dict
        """
        state = _init_loop_state(config)
        while _loop_should_continue(state, config):
            rewired = False
            for clique_size in config["allowed_clique_sizes"]:
                state["total_attempts"] += 1
                rewired = self._try_rewire_for_size(clique_size, config, state)
                if rewired or not _loop_should_continue(state, config):
                    break
        state["termination_reason"] = _get_termination_reason(state, config)
        return state

    def _try_rewire_for_size(
        self,
        clique_size: int,
        config: Dict[str, object],
        state: Dict[str, object],
    ):
        """
        Try one rewiring attempt for a specific candidate clique size.

        @param clique_size: Candidate clique size within the allowed band.
        @param config: Immutable run configuration.
        @param state: Mutable loop state.
        @return: True when a rewiring is accepted, otherwise False.
        """
        group = rng.sample(config["nodes"], clique_size)
        group_key = tuple(sorted(group))
        if group_key in state["checked_groups"]:
            state["cached_group_hits"] += 1
            state["consecutive_failures"] += 1
            return False

        outcome, missing_edges = self._evaluate_candidate(
            group,
            config,
            state["edge_count"],
        )
        self._apply_outcome(
            outcome,
            missing_edges,
            group_key,
            state,
            config["node_count"],
        )
        return outcome == OUTCOME_ACCEPTED

    def _evaluate_candidate(
        self,
        group: Sequence[int],
        config: Dict[str, object],
        edge_count: int,
    ):
        """
        Assess whether a candidate node group can form a valid new clique.

        @param group: Candidate nodes to form a clique.
        @param config: Immutable run configuration.
        @param edge_count: Current edge count for degree projection.
        @return: Tuple of (outcome_key, missing_edges). Outcome_key is one
            of 'accepted', 'full_clique_groups', or 'tolerance_rejections'.
        @rtype:  tuple[str, list]
        """
        missing_edges = [
            (u, v) for u, v in combinations(group, 2)
            if not self.ntwk.has_edge(u, v)
        ]
        if not missing_edges:
            return OUTCOME_FULL_CLIQUE, []
        new_avg_degree = (2 * (edge_count + len(missing_edges))) / config["node_count"]
        if _tolerance_exceeded(
            new_avg_degree, config["original_avg_degree"], config["degree_tolerance"]
        ):
            return OUTCOME_TOLERANCE_REJECTED, []
        return OUTCOME_ACCEPTED, missing_edges

    def _apply_outcome(
        self,
        outcome: str,
        missing_edges: List[Tuple[int, int]],
        group_key: Tuple[int, ...],
        state: Dict[str, object],
        node_count: int,
    ):
        """
        Mutate loop state based on the outcome of a candidate evaluation.

        @param outcome: 'accepted', 'full_clique_groups', or 'tolerance_rejections'.
        @param missing_edges: Edges to add when outcome is 'accepted'.
        @param group_key: Cache key for the candidate group.
        @param state: Mutable loop state (modified in-place).
        @param node_count: Node count for degree recalculation.
        """
        if outcome == OUTCOME_ACCEPTED:
            self.ntwk.add_edges_from(missing_edges)
            state["edge_count"] += len(missing_edges)
            state["final_avg_degree"] = (2 * state["edge_count"]) / node_count
            state["successful_rewirings"] += 1
            state["consecutive_failures"] = 0
            state["checked_groups"].clear()
        else:
            state[outcome] += 1
            state["checked_groups"].add(group_key)
            state["consecutive_failures"] += 1

    def _print_run_stats(self):
        """
        Print a formatted summary of the most recent rewiring run to stdout.
        """
        s = self.last_run_stats
        print("Rewiring stats:")
        print(f"  allowed clique sizes : {s['allowed_clique_sizes']}")
        print(f"  attempts            : {s['total_attempts']}")
        print(f"  successful rewires  : {s['successful_rewirings']}")
        print(f"  cache hits          : {s['cache_hits']}")
        print(f"  already-clique skips: {s['already_clique_groups']}")
        print(f"  tolerance rejections: {s['tolerance_rejections']}")
        print(
            f"  avg degree          : "
            f"{s['original_avg_degree']:.4f} -> {s['final_avg_degree']:.4f}"
        )
        print(f"  stop reason         : {s['termination_reason']}")

    def _validate_preconditions(self, target_clique_size: int):
        """
        Raise ValueError for any condition that would prevent rewiring.

        @param target_clique_size: Requested clique size.
        @raises ValueError: On disconnected graph, clique_size < 2, or
            clique_size > node count.
        """
        if not nx.is_connected(self.ntwk):
            raise ValueError("Graph is not connected.")
        if target_clique_size < 2:
            raise ValueError("target_clique_size must be at least 2.")
        if target_clique_size > self.ntwk.number_of_nodes():
            raise ValueError("target_clique_size cannot exceed the number of nodes.")

    def _build_allowed_clique_sizes(
        self,
        target_clique_size: int,
        node_count: int,
    ):
        """
        Build the allowed clique-size band, ordered by closeness to target.

        @param target_clique_size: Requested target clique size.
        @param node_count: Number of nodes available in the graph.
        @return: Ordered clique sizes within +/- 2 of the target.
        """
        min_size = max(2, target_clique_size - 2)
        max_size = min(node_count, target_clique_size + 2)
        return sorted(
            range(min_size, max_size + 1),
            key=lambda size: (abs(size - target_clique_size), size),
        )

    def _calculate_average_degree(self):
        """
        Compute the mean degree using the identity 2|E|/|V|.

        @return: Mean node degree, or 0.0 for an empty graph.
        @rtype:  float
        """
        node_count = self.ntwk.number_of_nodes()
        if node_count == 0:
            return 0.0
        return (2 * self.ntwk.number_of_edges()) / node_count



