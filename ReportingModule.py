"""
ReportingModule module.

Provides reusable network metrics and reporting helpers for the notebook.
"""
from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class ReportingModule:
    """
    Utility class for network metrics, visualisation, and run reporting.
    """

    @staticmethod
    def _calculate_average_degree(ntwk: nx.Graph):
        """Return mean node degree using 2|E|/|V|, or 0.0 for empty graphs."""
        node_count = ntwk.number_of_nodes()
        if node_count == 0:
            return 0.0
        return (2 * ntwk.number_of_edges()) / node_count

    def get_avg_degree_dist(self, ntwk: nx.Graph):
        """
        Compute the mean degree of all nodes in the network.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Mean node degree.
        @rtype:  float
        """
        return self._calculate_average_degree(ntwk)

    def get_average_degree(self, ntwk: nx.Graph):
        """Public alias with clearer naming for mean degree calculation."""
        return self.get_avg_degree_dist(ntwk)

    def _get_degree_counts(self, ntwk: nx.Graph):
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

    def get_degree_distribution(self, ntwk: nx.Graph):
        """
        Compute the degree sequence and per-degree counts of the network.

        @param ntwk: The network to analyse.
        @type  ntwk: nx.Graph
        @return: Tuple of (degree values array, count per degree array).
        @rtype:  tuple[np.ndarray, np.ndarray]
        """
        return self._get_degree_counts(ntwk)

    def _get_clique_size_counts(
        self,
        ntwk: nx.Graph,
        max_clique_size: int | None = None,
    ):
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

    @staticmethod
    def _normalise_counts(counts: np.ndarray):
        """Staticfunction to normalise a count array into a probability distribution.
        @param counts: Array of counts to normalise.
        @returns: Array of normalised values summing to 1.0, or zeros if input is empty.
        """
        total = counts.sum()
        if total == 0:
            return np.zeros_like(counts, dtype=float)
        return counts / total

    @staticmethod
    def _annotate_percentage_bars(ax, bars, values: np.ndarray):
        """Static method to annotate bar containers with percentage labels.

        @param ax: The matplotlib axes to annotate.
        @param bars: The bar containers to annotate.
        @param values: The values corresponding to each bar.
        """
        if values.size == 0:
            return

        max_value = float(values.max()) if values.size > 0 else 0.0
        # Ensure tiny bars still get visible labels above the baseline.
        min_offset = 0.002
        relative_offset = max_value * 0.015
        y_offset = max(min_offset, relative_offset)

        for bar, value in zip(bars, values):
            x = bar.get_x() + (bar.get_width() / 2)
            if value <= 0:
                ax.text(
                    x,
                    y_offset,
                    "No Value",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    alpha=0.8,
                )
                continue
            y = bar.get_height()
            ax.text(x, y + y_offset, f"{value:.2%}", ha="center", va="bottom", fontsize=8)

    @staticmethod
    def _aligned_degree_series(
        before_degrees: np.ndarray,
        before_counts: np.ndarray,
        after_degrees: np.ndarray,
        after_counts: np.ndarray,
    ):
        """Static function to align degree count series onto the same x-axis for comparison plots.

        @param before_degrees: Degrees in the 'before' network.
        @param before_counts: Counts corresponding to each degree in the 'before' network.
        @param after_degrees: Degrees in the 'after' network.
        @param after_counts: Counts corresponding to each degree in the 'after' network.
        @returns: Tuple of (aligned_degrees, before_aligned, after_aligned)
        """
        max_degree = -1
        if before_degrees.size > 0:
            max_degree = max(max_degree, int(before_degrees.max()))
        if after_degrees.size > 0:
            max_degree = max(max_degree, int(after_degrees.max()))
        if max_degree < 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

        aligned_degrees = np.arange(max_degree + 1)
        before_aligned = np.zeros(max_degree + 1, dtype=int)
        after_aligned = np.zeros(max_degree + 1, dtype=int)

        if before_degrees.size > 0:
            before_aligned[before_degrees] = before_counts
        if after_degrees.size > 0:
            after_aligned[after_degrees] = after_counts

        return aligned_degrees, before_aligned, after_aligned

    def visualise_before_after_comparison(
        self,
        before_ntwk: nx.Graph,
        after_ntwk: nx.Graph,
        max_clique_size: int | None = None,
        title_prefix: str = "",
    ):
        """
        Plot compact before-vs-after degree and clique distributions.

        @param before_ntwk: Network state before rewiring.
        @param after_ntwk: Network state after rewiring.
        @param max_clique_size: Optional cap for reported clique sizes.
        @param title_prefix: Optional text prefix in subplot titles.
        """
        before_degrees, before_degree_counts = self.get_degree_distribution(before_ntwk)
        after_degrees, after_degree_counts = self.get_degree_distribution(after_ntwk)

        degrees, before_aligned, after_aligned = self._aligned_degree_series(
            before_degrees,
            before_degree_counts,
            after_degrees,
            after_degree_counts,
        )

        before_clique_counts = self._get_clique_size_counts(before_ntwk, max_clique_size)
        after_clique_counts = self._get_clique_size_counts(after_ntwk, max_clique_size)
        clique_sizes = np.array(
            sorted(set(before_clique_counts.keys()) | set(after_clique_counts.keys())),
            dtype=int,
        )

        if clique_sizes.size > 0:
            before_clique_values = np.array(
                [before_clique_counts.get(size, 0) for size in clique_sizes],
                dtype=float,
            )
            after_clique_values = np.array(
                [after_clique_counts.get(size, 0) for size in clique_sizes],
                dtype=float,
            )
            before_clique_norm = self._normalise_counts(before_clique_values)
            after_clique_norm = self._normalise_counts(after_clique_values)
        else:
            before_clique_norm = np.array([], dtype=float)
            after_clique_norm = np.array([], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        degree_ax, clique_ax = axes

        if degrees.size == 0:
            degree_ax.text(0.5, 0.5, "No degree data", ha="center", va="center")
            degree_ax.set_axis_off()
        else:
            degree_ax.step(degrees, before_aligned, where="mid", linewidth=1.8, label="Before")
            degree_ax.step(degrees, after_aligned, where="mid", linewidth=1.8, label="After")
            degree_ax.set_title(f"{title_prefix}Degree Distribution Comparison")
            degree_ax.set_xlabel("Degree")
            degree_ax.set_ylabel("Count")
            degree_ax.grid(axis="y", linestyle="--", alpha=0.3)
            degree_ax.legend()

        if clique_sizes.size == 0:
            clique_ax.text(0.5, 0.5, "No clique data", ha="center", va="center")
            clique_ax.set_axis_off()
        else:
            bar_width = 0.38
            before_bars = clique_ax.bar(
                clique_sizes - (bar_width / 2),
                before_clique_norm,
                width=bar_width,
                label="Before",
                alpha=0.9,
            )
            after_bars = clique_ax.bar(
                clique_sizes + (bar_width / 2),
                after_clique_norm,
                width=bar_width,
                label="After",
                alpha=0.9,
            )
            self._annotate_percentage_bars(clique_ax, before_bars, before_clique_norm)
            self._annotate_percentage_bars(clique_ax, after_bars, after_clique_norm)
            clique_ax.set_title(f"{title_prefix}Clique Size Comparison (Normalised)")
            clique_ax.set_xlabel("Clique Size")
            clique_ax.set_ylabel("Proportion of Cliques")
            clique_ax.set_xticks(clique_sizes)
            clique_ax.grid(axis="y", linestyle="--", alpha=0.3)
            clique_ax.legend()

            if clique_sizes.size > 12:
                clique_ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def visualise_cliques_distribution(
        self,
        ntwk: nx.Graph,
        max_clique_size: int | None = None,
    ):
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

    def visualise_degree_distribution(self, ntwk: nx.Graph):
        """
        Plot the degree distribution of the network as a bar chart.

        @param ntwk: The network to visualise.
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

    @staticmethod
    def _print_network_snapshot(label: str, ntwk: nx.Graph, avg_degree: float):
        """Static Method to print a compact textual summary for a network state.
        @param label: Descriptive heading for this snapshot.
        @param ntwk: The network to summarise.
        @param avg_degree: The average degree of the network.
        """
        print(label)
        print(f"Total nodes: {ntwk.number_of_nodes()}")
        print(f"Total edges: {ntwk.number_of_edges()}")
        print(f"Average degree: {avg_degree}")

    def log_network_state(self, label: str, ntwk: nx.Graph):
        """
        Print a labelled snapshot of a network: mean degree, degree plot,
        and clique-size distribution.

        @param label: Descriptive heading for this snapshot.
        @param ntwk: The network to log.
        """
        avg_degree = self.get_average_degree(ntwk)
        self._print_network_snapshot(label, ntwk, avg_degree)
        self.visualise_degree_distribution(ntwk)
        self.visualise_cliques_distribution(ntwk)
