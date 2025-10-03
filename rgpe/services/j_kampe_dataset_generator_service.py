import mpmath as mp
import pandas as pd
from . import dataset_loader_service
from . import riemann_service
from ..utils import get_lagged_dataframe


def add_lags_to_j_kampe_dataset(original_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([original_df, get_lagged_dataframe(original_df["gram"], 10, "gram")], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["z_gram"], 10, "z_gram")], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["d"], 25, "d")], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["cogram"], 10, "cogram")], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["z_cogram"], 15, "z_cogram")], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["z_integer"], 10, "z_integer")], axis=1)
    return df


def generate_j_kampe_dataset(limit: int = 100_000) -> None:
    print("Generating J Kampee dataset...")
    gram_points = dataset_loader_service.load_gram_points()
    distances   = dataset_loader_service.load_distances()
    cogram_points = dataset_loader_service.load_cogram_points()
    rows = []
    counter = 0

    for i, (gram, cogram, d) in enumerate(zip(gram_points, cogram_points, distances)):
        row = {
            "gram": gram,
            "cogram": cogram,
            "d": d,
            "z_gram": mp.siegelz(gram),
            "z_cogram": mp.siegelz(cogram),
            "z_integer": mp.siegelz(i + 1)
        }
        row.update(riemann_service.get_Z_function_terms_features(gram))
        rows.append(row)
        print(f"i: {i} | Gram: {gram} | Cogram: {cogram} | Distance: {d}")
        counter += 1
        if counter >= limit:
            break

    df = pd.DataFrame(rows)
    df = add_lags_to_j_kampe_dataset(df)
    df = df.drop(columns=["d"])
    df.fillna(0).to_csv("/app/dataset/j_kampe.csv", index=False)
    print(f"[OK] Dataset gerado com shape: {df.shape}\n")
