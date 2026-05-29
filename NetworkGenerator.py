
"""
NetworkGenerator module.

Provides factory-style generation of Erdos-Renyi random graphs
with optional connectivity guarantees.
"""
import networkx as nx


class NetworkGenerator:
    """
    Factory class for generating Erdos-Renyi (ER) random graphs.

    @param num_nodes: Number of nodes in the graph.
    @type  num_nodes: int
    @param kmean: Target mean degree (expected number of connections per node).
    @type  kmean: float
    """

    def __init__(self, num_nodes, kmean):
        """
        Initialise the generator and derive the edge probability from kmean.

        @param num_nodes: Number of nodes in the graph.
        @type  num_nodes: int
        @param kmean: Target mean degree.
        @type  kmean: float
        """
        self.N = num_nodes
        self.kmean = kmean
        self.p = kmean / num_nodes

    def generate_erdos_renyi_graph(self, require_connected=False, max_attempts=100):
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
        if not require_connected:
            return nx.erdos_renyi_graph(self.N, self.p)

        for _ in range(max_attempts):
            ntwk = nx.erdos_renyi_graph(self.N, self.p)
            if nx.is_connected(ntwk):
                return ntwk

        raise ValueError(
            "Could not generate a connected ER graph. "
            "Try a higher kmean or more attempts."
        )
