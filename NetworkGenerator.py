
"""
NetworkGenerator module.

Provides factory-style generation of Erdos-Renyi random graphs
with optional connectivity guarantees.
"""
from __future__ import annotations

import math
import networkx as nx


class NetworkGenerator:
    """
    Class for generating Erdos-Renyi (ER) random graphs.

    @param num_nodes: Number of nodes in the graph.
    @param kmean: Target mean degree (expected number of connections per node).
    """

    def __init__(self, num_nodes: int, kmean: float):
        """
        Method to initialise the generator and derive edge probability from kmean.

        @param num_nodes: Number of nodes in the graph.
        @param kmean: Target mean degree.
        """
        self._validate_inputs(num_nodes, kmean)
        self.num_nodes = num_nodes
        self.kmean = kmean
        self.edge_probability = self._calculate_edge_probability(num_nodes, kmean)

        # Backward-compatible aliases retained for notebook code that may
        # access these attributes directly.
        self.N = self.num_nodes
        self.p = self.edge_probability

    @staticmethod
    def _validate_inputs(num_nodes: int, kmean: float):
        """Static method to validate constructor inputs for sensible ER graph generation.
        @param num_nodes: Number of nodes in the graph.
        @param kmean: Target mean degree.
        @raises ValueError: If num_nodes <= 0 or kmean < 0.
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be greater than 0.")
        if kmean < 0:
            raise ValueError("kmean must be non-negative.")

    @staticmethod
    def _calculate_edge_probability(num_nodes: int, kmean: float):
        """Static method to calculate ER edge probability from node count and mean degree.
        @param num_nodes: Number of nodes in the graph.
        @param kmean: Target mean degree.
        @return: Edge probability for the ER graph.
        """
        if num_nodes <= 1:
            return 0.0
        return kmean / (num_nodes - 1)

    def generate_erdos_renyi_graph(
        self,
        require_connected: bool = False,
        max_attempts: int = 100,
    ):
        """
        Method to generate an Erdos-Renyi G(N, p) random graph.

        When require_connected is True, the method retries until a connected
        graph is produced or max_attempts is exhausted.

        @param require_connected: Whether the returned graph must be connected.
        @param max_attempts: Maximum generation attempts when connectivity is required.
        @return: A random Erdos-Renyi graph.
        @raises ValueError: If a connected graph cannot be produced within max_attempts tries.
        """
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0.")

        if not require_connected:
            return nx.erdos_renyi_graph(self.num_nodes, self.edge_probability)

        # For large ER graphs, connectivity generally needs mean degree around
        # log(n). Start from a connectivity-aware k and nudge it per attempt.
        min_connected_k = math.log(self.num_nodes) + 1.5
        starting_k = max(self.kmean, min_connected_k)

        for attempt in range(max_attempts):
            attempt_k = starting_k + (0.05 * attempt)
            attempt_p = min(1.0, attempt_k / (self.num_nodes - 1))
            ntwk = nx.erdos_renyi_graph(self.num_nodes, attempt_p)
            if nx.is_connected(ntwk):
                return ntwk

        raise ValueError(
            "Could not generate a connected ER graph after "
            f"{max_attempts} attempts (num_nodes={self.num_nodes}, "
            f"kmean={self.kmean}). Try a higher kmean or more attempts."
        )
