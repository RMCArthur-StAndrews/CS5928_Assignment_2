"""
ReportingModule module.

Provides reusable network metrics and reporting helpers for the notebook.
"""
import gc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


class ReportingModule:
    """
    Utility class for network metrics, visualisation, and run reporting.
    """

    def get_avg_degree_dist(self, ntwk):
        """
        Compute the mean degree of all nodes in the network.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Mean node degree.
        @rtype:  float
        """
        node_count = ntwk.number_of_nodes()
        if node_count == 0:
            return 0.0
        return (2 * ntwk.number_of_edges()) / node_count

    def _get_degree_counts(self, ntwk):
        """
        Build a frequency distribution for node degrees.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Tuple of (degree values array, count per degree array).
        @rtype:  tuple[np.ndarray, np.ndarray]
        """
        degree_sequence = np.fromiter((d for _, d in ntwk.degree()), dtype=int)
        if degree_sequence.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        degree_counts = np.bincount(degree_sequence)
        degrees = np.arange(degree_counts.size)
        return degrees, degree_counts

    def get_degree_distribution(self, ntwk):
        """
        Compute the degree sequence and per-degree counts of the network.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Tuple of (degree values array, count per degree array).
        @rtype:  tuple[np.ndarray, np.ndarray]
        """
        return self._get_degree_counts(ntwk)

    def _get_clique_size_counts(self, ntwk, max_clique_size=None):
        """
        Count maximal cliques by size.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @param max_clique_size: Optional upper bound on clique sizes to report
        @type  max_clique_size: int or None. Contains the max allowed clique size
        @return: Counter keyed by clique size.
        @rtype:  Counter
        """
        clique_size_counts = Counter()
        for clique in nx.find_cliques(ntwk):
            clique_size = len(clique)
            if max_clique_size is None or clique_size <= max_clique_size:
                clique_size_counts[clique_size] += 1
        return clique_size_counts

    def visualise_cliques_distribution(self, ntwk, max_clique_size=None):
        """
        Print and plot the clique-size distribution of the network.

        Raw clique counts are printed in text form. The graph view is normalized
        so sizes with very different frequencies remain readable.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @param max_clique_size: Optional upper bound on clique sizes to report.
        @type  max_clique_size: int or None. Contains the max allowed clique size
        """
        clique_size_counts = self._get_clique_size_counts(ntwk, max_clique_size)

        if not clique_size_counts:
            print("No cliques found.")
            return

        print("Clique size distribution (raw counts):")
        for size, count in sorted(clique_size_counts.items()):
            print(f"  Size {size}: {count} clique(s)")

        sizes = np.array(sorted(clique_size_counts.keys()))
        counts = np.array([clique_size_counts[size] for size in sizes], dtype=float)
        total = counts.sum()
        if total == 0:
            print("No clique counts available for visualization.")
            return
        normalized_counts = counts / total

        print(
            "Clique summary: "
            f"total maximal cliques={int(total)}, "
            f"dominant size={sizes[np.argmax(counts)]}"
        )
        print("Clique sizes present: " + ", ".join(map(str, sizes)))

        fig, ax = plt.subplots(figsize=(12, 5))

        bars = ax.bar(
            sizes,
            normalized_counts,
            color="#9ecae1",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )

        if len(sizes) <= 10:
            annotated_indexes = range(len(sizes))
        else:
            annotated_indexes = np.argsort(normalized_counts)[-8:]

        for index in annotated_indexes:
            x = sizes[index]
            y = normalized_counts[index]
            ax.text(x, y + 0.002, f"{y:.2%}", ha="center", va="bottom", fontsize=9)

        ax.set_title("Normalized Clique Size Distribution (Bar Plot)")
        ax.set_xlabel("Clique Size")
        ax.set_ylabel("Proportion of Cliques")
        ax.set_xticks(sizes)
        ax.set_ylim(0, max(normalized_counts) * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if len(sizes) > 12:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
        plt.close(fig)
        del clique_size_counts, sizes, counts, normalized_counts, bars
        gc.collect()

    def visualise_degree_distribution(self, ntwk):
        """
        Plot the degree distribution of the network as a bar chart.

        @param ntwk: The network to visualise.
        @type  ntwk: nx.Graph
        """
        degrees, degree_counts = self.get_degree_distribution(ntwk)
        if degree_counts.size == 0:
            print("No degree distribution available.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        if degree_counts.size <= 60:
            ax.bar(degrees, degree_counts, width=0.80, color="#3182bd", alpha=0.9)
        else:
            ax.step(degrees, degree_counts, where="mid", color="#08519c", linewidth=1.8)
            ax.fill_between(degrees, degree_counts, step="mid", alpha=0.2, color="#6baed6")
        ax.set_title("Degree Distribution")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        del degrees, degree_counts
        gc.collect()

    def log_network_state(self, label, ntwk):
        """
        Print a labelled snapshot of a network: mean degree, degree plot,
        and clique-size distribution.

        @param label: Descriptive heading for this snapshot.
        @param ntwk: The network to log.
        """
        print(label)
        print("Total nodes: " + str(ntwk.number_of_nodes()))
        print("Total edges: " + str(ntwk.number_of_edges()))
        print("Average degree: " + str(self.get_avg_degree_dist(ntwk)))
        self.visualise_degree_distribution(ntwk)
        self.visualise_cliques_distribution(ntwk)
