import requests
import mpmath as mp
import numpy as np
import pandas as pd
from . import dataset_loader_service
from . import riemann_service


def generate_gram_points_dataset(start: int = 0, end: int = 100_000, seed: float = 7.0) -> None:
    """
    Gera um CSV com Gram points de start até end.
    Usa o Gram point anterior como chute inicial para o próximo.
    """
    print("Generating Gram Points...")

    t0 = mp.mpf(seed)
    gram_points = []

    for n in range(start, end):
        gram_point = riemann_service.get_gram_point(n - 1, t0)
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
        gram_point = riemann_service.get_cogram_point(n - 1, t0)
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
