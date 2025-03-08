#graph_builder.py
import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def build_graph_from_dataframe(df: pd.DataFrame, source_col: str, target_col: str) -> Data:
    """
    Build a graph from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the edges.
        source_col (str): The name of the column containing the source nodes.
        target_col (str): The name of the column containing the target nodes.

    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
    data = from_networkx(G)
    return data