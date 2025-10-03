import mpmath as mp
import pandas as pd
from typing import List
from . import dataset_loader_service


def _get_o_shank_dataset_header() -> List[str]:
    header = []
    for n in [1, 2]:
        header.append(f"z_value_{n}")
        header += [f"z_cos_term_{i}_{n}" for i in range(1, 11)]
        header += [f"z_sin_term_{i}_{n}" for i in range(2, 11)]
    return header


def _get_o_shank_features(gram_point: float) -> List[float]:
    features = [mp.siegelz(gram_point)]
    cos_terms = [mp.cos(mp.siegeltheta(gram_point) - gram_point * mp.ln(n)) / mp.sqrt(n) for n in range(1, 11)]
    sin_terms = [mp.sin(mp.siegeltheta(gram_point) - gram_point * mp.ln(n)) / mp.sqrt(n) for n in range(2, 11)]
    return features + cos_terms + sin_terms


def generate_o_shank_dataset() -> None:
    gram_points = dataset_loader_service.load_gram_points()
    features = []

    NUMBER_OF_POINTS = 10_000

    header = _get_o_shank_dataset_header()

    print("Generating O-Shank dataset...")

    for index in range(NUMBER_OF_POINTS):
        gram_point_1 = gram_points[index - 1]
        gram_point_2 = gram_points[index]
        features_1 = _get_o_shank_features(gram_point_1)
        features_2 = _get_o_shank_features(gram_point_2)
        features.append(features_1 + features_2)

    df_features = pd.DataFrame(features, columns=header)
    df_features.to_csv("/app/dataset/o_shank.csv", index=False)
    print(f"[OK] Dataset gerado com shape: {df_features.shape}\n")
