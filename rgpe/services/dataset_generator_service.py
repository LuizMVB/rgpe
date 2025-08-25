import requests
import numpy as np
import pandas as pd
from ..lib.gram_points.GramPoints import write_gram_points


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
    Gera as disntâncias entre o zero e o ponto de gram.
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
