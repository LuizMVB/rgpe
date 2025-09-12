import requests
import mpmath as mp
import numpy as np
import pandas as pd
from typing import List
from ..lib.gram_points.GramPoints import write_gram_points
from . import dataset_loader_service


def generate_gram_points_dataset() -> None:
    write_gram_points('/app/dataset/gram_points.csv', 0, 100000)


def download_zeta_zeros() -> None:
    """
    Faz download dos zeros da função zeta de Riemann e salva em CSV.
    """
    url = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1"
    r = requests.get(url)
    r.raise_for_status()
    zeros = np.fromstring(r.text, sep="\n").astype(float)
    df = pd.DataFrame({"zeta_zero": zeros})
    df.to_csv("/app/dataset/zeta_zeros.csv", index=False)
    print(f"[OK] Arquivo salvo: dataset/zeta_zeros.csv com {len(zeros)} zeros.")


def write_distances_dataset() -> None:
    """
    Gera as distâncias entre o zero e o ponto de gram.
    """
    df_zeros = pd.read_csv("/app/dataset/zeta_zeros.csv")
    df_gram  = pd.read_csv("/app/dataset/gram_points.csv")

    zeros = df_zeros["zeta_zero"].values
    gram_points = df_gram["n-th gram point"].values

    y = zeros - gram_points

    df = pd.DataFrame({"distance": y})
    df.to_csv("/app/dataset/distances.csv", index=False)
    print("[OK] Dataset gerado com shape:", df.shape)


def generate_distances_dataset() -> None:
    download_zeta_zeros()
    write_distances_dataset()


def _get_header() -> List[str]:
    header = []
    for n in [1, 2]:
        header.append(f"z_value_{n}")
        header += [f"z_cos_{i}_{n}" for i in range(1, 11)]
        header += [f"z_sin_{i}_{n}" for i in range(2, 11)]
    return header


def _get_features(gram_point: float) -> List[float]:
    features = [mp.siegelz(gram_point)]
    cos_terms = [mp.cos(mp.siegeltheta(gram_point) - gram_point * mp.ln(n)) / mp.sqrt(n) for n in range(1, 11)]
    sin_terms = [mp.sin(mp.siegeltheta(gram_point) - gram_point * mp.ln(n)) / mp.sqrt(n) for n in range(2, 11)]
    return features + cos_terms + sin_terms


def generate_40_features_dataset() -> None:
    gram_points = dataset_loader_service.load_gram_points()
    features = []

    header = _get_header()

    for index in range(1, len(gram_points)):
        gram_point_1 = gram_points[index - 1]
        gram_point_2 = gram_points[index]

        print(f"Gerando para {gram_point_1} e {gram_point_2}")

        features_1 = _get_features(gram_point_1)
        features_2 = _get_features(gram_point_2)
        features.append(features_1 + features_2)

    df_features = pd.DataFrame(features, columns=header)
    df_features.to_csv("/app/dataset/o_shank.csv", index=False)
