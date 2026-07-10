
"""
NetworkGenerator module.

Provides factory-style generation of Erdos-Renyi random graphs
with optional connectivity guarantees.
"""
from __future__ import annotations

import networkx as nx


class NetworkGenerator:
    """
    Factory class for generating Erdos-Renyi (ER) random graphs.

    @param num_nodes: Number of nodes in the graph.
    @type  num_nodes: int
    @param kmean: Target mean degree (expected number of connections per node).
    @type  kmean: float
    """

    def __init__(self, num_nodes: int, kmean: float):
        """
        Initialise the generator and derive the edge probability from kmean.

        @param num_nodes: Number of nodes in the graph.
        @type  num_nodes: int
        @param kmean: Target mean degree.
        @type  kmean: float
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
        """Validate constructor inputs for sensible ER graph generation."""
        if num_nodes <= 0:
            raise ValueError("num_nodes must be greater than 0.")
        if kmean < 0:
            raise ValueError("kmean must be non-negative.")

    @staticmethod
    def _calculate_edge_probability(num_nodes: int, kmean: float):
        """Return the ER edge probability derived from kmean and node count."""
        return kmean / num_nodes

    def generate_erdos_renyi_graph(
        self,
        require_connected: bool = False,
        max_attempts: int = 100,
    ):
        """
        Generate an Erdos-Renyi G(N, p) random graph.

        When *require_connected* is True the method retries until a connected
        graph is produced or *max_attempts* is exhausted.

        @param require_connected: Whether the returned graph must be connected.
        @type  require_connected: bool
        @param max_attempts: Maximum generation attempts when connectivity is
            required.
        @type  max_attempts: int
        @return: A random Erdos-Renyi graph.
        @rtype:  nx.Graph
        @raises ValueError: If a connected graph cannot be produced within
            *max_attempts* tries.
        """
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0.")

        if not require_connected:
            return nx.erdos_renyi_graph(self.num_nodes, self.edge_probability)

        for _ in range(max_attempts):
            ntwk = nx.erdos_renyi_graph(self.num_nodes, self.edge_probability)
            if nx.is_connected(ntwk):
                return ntwk

        raise ValueError(
            "Could not generate a connected ER graph. "
            "Try a higher kmean or more attempts."
        )
