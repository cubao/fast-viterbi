"""

Pybind11 example plugin
-----------------------

.. currentmodule:: scikit_build_example

.. autosummary::
   :toctree: _generate

   add
   subtract

"""

from __future__ import annotations

import typing

__all__ = ["FastViterbi", "add", "subtract"]

class FastViterbi:
    def __init__(
        self,
        K: int,
        N: int,
        scores: dict[tuple[tuple[int, int], tuple[int, int]], float],
    ) -> None:
        """
        Initialize FastViterbi object.

        Args:
            K (int): Number of nodes per layer.
            N (int): Number of layers.
            scores (dict): Scores for node transitions.
        """
    def all_road_paths(self) -> list[list[int]]:
        """
        Get all road paths.

        Returns:
            list: All road paths in the graph.
        """
    @typing.overload
    def inference(self) -> tuple[float, list[int]]:
        """
        Perform inference without a road path.

        Returns:
            tuple: Best path and its score.
        """
    @typing.overload
    def inference(self, road_path: list[int]) -> tuple[float, list[int], list[int]]:
        """
        Perform inference with a given road path.

        Args:
            road_path (list): List of road indices representing a path.

        Returns:
            tuple: Best path and its score.
        """
    def scores(self, node_path: list[int]) -> list[float]:
        """
        Get scores for a given node path.

        Args:
            node_path (list): List of node indices representing a path.

        Returns:
            float: Total score for the given path.
        """
    def setup_roads(self, roads: list[list[int]]) -> bool:
        """
        Set up roads for the Viterbi algorithm.

        Args:
            roads (list): List of road sequences.
        """
    def setup_shortest_road_paths(
        self, sp_paths: dict[tuple[tuple[int, int], tuple[int, int]], list[int]]
    ) -> bool:
        """
        Set up shortest road paths.

        Args:
            sp_paths (dict): Dictionary of shortest paths between nodes.
        """

def add(arg0: int, arg1: int) -> int:
    """
    Add two numbers

    Some other explanation about the add function.
    """

def subtract(arg0: int, arg1: int) -> int:
    """
    Subtract two numbers

    Some other explanation about the subtract function.
    """

__version__: str = "0.1.2"
