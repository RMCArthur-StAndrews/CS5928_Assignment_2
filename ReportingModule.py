"""
ReportingModule module.

Provides reusable network metrics and reporting helpers for the notebook.
"""
import gc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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
        degrees = [d for _, d in ntwk.degree()]
        return sum(degrees) / len(degrees)

    def get_degree_distribution(self, ntwk):
        """
        Compute the degree sequence and per-degree counts of the network.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Tuple of (degree values array, count per degree array).
        @rtype:  tuple[np.ndarray, np.ndarray]
        """
        degree_sequence = sorted([d for _, d in ntwk.degree()], reverse=True)
        degree_counts = np.bincount(degree_sequence)
        degrees = np.arange(len(degree_counts))
        return degrees, degree_counts

    def visualise_cliques_distribution(self, ntwk, max_clique_size=None):
        """
        Print and plot the clique-size distribution of the network.

        Raw clique counts are printed in text form. The graph view is normalized
        so sizes with very different frequencies remain readable.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @param max_clique_size: Optional upper bound on clique sizes to report.
        @type  max_clique_size: int or None
        """
        cliques = (sorted(c) for c in nx.find_cliques(ntwk))
        if max_clique_size is not None:
            cliques = (c for c in cliques if len(c) <= max_clique_size)

        clique_size_counts = {}
        for clique in cliques:
            size = len(clique)
            clique_size_counts[size] = clique_size_counts.get(size, 0) + 1

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

        for x, y in zip(sizes, normalized_counts):
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
        fig, ax = plt.subplots()
        ax.bar(degrees, degree_counts, width=0.80, color="b")
        ax.set_title("Degree Distribution")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
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
        print("Average degree: " + str(self.get_avg_degree_dist(ntwk)))
        self.visualise_degree_distribution(ntwk)
        self.visualise_cliques_distribution(ntwk)
