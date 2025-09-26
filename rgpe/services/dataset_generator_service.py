import requests
import mpmath as mp
import numpy as np
import pandas as pd
from typing import List
from . import dataset_loader_service


def _get_gram_point(n: int, t0: float) -> float:
    """
    Calcula o n-ésimo Gram point usando t0 como chute inicial.
    θ(t) = n * π
    """
    f = lambda t: mp.siegeltheta(t) - n * mp.pi
    return mp.findroot(f, t0)


def _get_cogram_point(n: int, t0: float) -> float:
    """Resolve θ(t) = (n+1/2)π perto de t0 (Gram point g_n usado como chute)."""
    f = lambda t: mp.siegeltheta(t) - (n + 0.5) * mp.pi
    return mp.findroot(f, t0)


def generate_gram_points_dataset(start: int = 0, end: int = 100_000, seed: float = 7.0) -> None:
    """
    Gera um CSV com Gram points de start até end.
    Usa o Gram point anterior como chute inicial para o próximo.
    """
    print("Generating Gram Points...")

    t0 = mp.mpf(seed)
    gram_points = []

    for n in range(start, end):
        gram_point = _get_gram_point(n - 1, t0)
        gram_points.append((n, gram_point))
        t0 = gram_point

        if n % 1000 == 0:
            print(f"Gram Progress: {n / end * 100:.2f}%")

    df = pd.DataFrame(gram_points, columns=["n", "gram_point"])
    df.to_csv("/app/dataset/gram_points.csv", index=False)
    print("Done.\n")


def generate_cogram_points_dataset(start: int = 0, end: int = 100_000, seed: float = 7.0) -> None:
    print("Generating coGram Points...")

    t0 = mp.mpf(seed)
    gram_points = []

    for n in range(start, end):
        gram_point = _get_cogram_point(n - 1, t0)
        gram_points.append((n, gram_point))
        t0 = gram_point

        if n % 1000 == 0:
            print(f"coGram Progress: {n / end * 100:.2f}%")

    df = pd.DataFrame(gram_points, columns=["n", "cogram_point"])
    df.to_csv("/app/dataset/cogram_points.csv", index=False)
    print("Done.\n")


def _download_zeta_zeros() -> None:
    """
    Faz download dos zeros da função zeta de Riemann e salva em CSV.
    """
    print("Downloading zeta zeros...")
    url = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1"
    r = requests.get(url)
    r.raise_for_status()
    zeros = np.fromstring(r.text, sep="\n").astype(float)
    df = pd.DataFrame({"zeta_zero": zeros})
    df.to_csv("/app/dataset/zeta_zeros.csv", index=False)
    print(f"[OK] Arquivo salvo: dataset/zeta_zeros.csv com {len(zeros)} zeros.")


def _write_distances_dataset() -> None:
    """
    Gera as distâncias entre o zero e o ponto de gram.
    """
    print("Generating distances dataset...")

    zeros = dataset_loader_service.load_zeta_zeros()
    gram_points = dataset_loader_service.load_gram_points()

    y = zeros - gram_points

    df = pd.DataFrame({"distance": y})
    df.to_csv("/app/dataset/distances.csv", index=False)
    print(f"[OK] Dataset gerado com shape: {df.shape}\n")


def generate_distances_dataset() -> None:
    _download_zeta_zeros()
    _write_distances_dataset()


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


def _get_Z_function_terms_features(t: float, max_term: int = 10):
    """
    Obtém os termo de 2 até 10 da função Z
    """
    theta = mp.siegeltheta(t)
    out = {}
    for n in range(2, max_term+1):
        angle = theta - t * mp.ln(n)
        term = 2 * mp.cos(angle) / mp.sqrt(n)
        out[f"z_term_{n}"] = term
    return out


def _lagged(series: pd.Series, max_lag: int, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for k in range(1, max_lag + 1):
        df[f"{prefix}_lag_{k}"] = pd.Series(series).shift(k)
    return df


def _add_lags(df: pd.DataFrame):
    df = pd.concat([df, _lagged(df["gram"], 10, "gram")], axis=1)
    df = pd.concat([df, _lagged(df["z_gram"], 10, "z_gram")], axis=1)
    df = pd.concat([df, _lagged(df["d"], 25, "d")], axis=1)
    df = pd.concat([df, _lagged(df["cogram"], 10, "cogram")], axis=1)
    df = pd.concat([df, _lagged(df["z_cogram"], 15, "z_cogram")], axis=1)
    df = pd.concat([df, _lagged(df["z_integer"], 10, "z_integer")], axis=1)


def generate_j_kampe_dataset() -> None:
    print("Generating J Kampee dataset...")
    gram_points = dataset_loader_service.load_gram_points()
    distances   = dataset_loader_service.load_distances()
    cogram_points = dataset_loader_service.load_cogram_points()

    n_points = len(gram_points)
    rows = []

    for i, (gram, cogram, d) in enumerate(zip(gram_points, cogram_points, distances)):
        row = {
            "index": i,
            "gram": gram,
            "cogram": cogram,
            "distance": d,
            "z_gram": mp.siegelz(gram),
            "z_cogram": mp.siegeltheta(gram),
        }
        row.update(_get_Z_function_terms_features(gram))
        row["z_integer"] = float(mp.siegelz(int(np.floor(gram))))
        rows.append(row)
        print(f"i: {i} | Gram: {gram} | Cogram: {cogram} | Distance: {d}")

    df = pd.DataFrame(rows)
    _add_lags(df)
    df.to_csv("/app/dataset/j_kampe.csv", index=False)
    print(f"[OK] Dataset gerado com shape: {df.shape}\n")
