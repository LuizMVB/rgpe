import pandas as pd
import numpy as np
from numpy.typing import NDArray


def load_gram_points() -> NDArray:
    gram_points_df = pd.read_csv("/app/dataset/gram_points.csv")
    return gram_points_df["n-th gram point"].values


def load_gram_distance_dataset() -> tuple[NDArray[float], NDArray[float]]:
    """
    Load the Gram distance dataset from a CSV file.
    
    Returns:
        X (np.ndarray): Features of the dataset.
        y (np.ndarray): Target values of the dataset.
    """
    df = pd.read_csv("/app/dataset/gram_distance.csv")
    X = df[["gram_point"]].values
    y = df["distance_to_zero"].values
    X = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)  # Ensure X is 2D
    return X, y


def load_40_features_dataset() -> tuple[NDArray[float], NDArray[float]]:
    df_features = pd.read_csv("/app/dataset/40_features.csv")
    df_distances = pd.read_csv("/app/dataset/distances.csv")
    df_distances = df_distances[:df_features.shape[0]]
    X = df_features.values
    y = df_distances.iloc[:, 0].values
    return X, y
