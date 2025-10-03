import pandas as pd
import numpy as np
import mpmath as mp
from . import dataset_loader_service, riemann_service
from ..utils import get_lagged_dataframe
from .j_kampe_dataset_generator_service import add_lags_to_j_kampe_dataset


def _add_lags_to_custom_dataset(original_df: pd.DataFrame) -> pd.DataFrame:
    df = add_lags_to_j_kampe_dataset(original_df)
    df = pd.concat([df, get_lagged_dataframe(df["gram"], 200, "gram", 20, 10)], axis=1)
    df = pd.concat([df, get_lagged_dataframe(df["z_gram"], 200, "z_gram", 20, 10)], axis=1)
    return df


def _append_first_difference_feature(df: pd.DataFrame) -> None:
    max_first_lags = 10
    first_lags_step = 1
    max_second_lags = 200
    second_lags_step = 10

    for i in range(1, max_first_lags):
        df[f"gram_first_difference_{i}"] = df[f"gram_lag_{i}"] - df[f"gram_lag_{i + first_lags_step}"]
        df[f"z_gram_first_difference_{i}"] = df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + first_lags_step}"]

    for i in range(20, max_second_lags, second_lags_step):
        df[f"gram_first_difference_{i}"] = df[f"gram_lag_{i}"] - df[f"gram_lag_{i + second_lags_step}"]
        df[f"z_gram_first_difference_{i}"] = df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + second_lags_step}"]


def _append_second_difference_feature(df: pd.DataFrame) -> None:
    max_first_lags = 9
    first_lags_step = 1
    max_second_lags = 190
    second_lags_step = 10

    for i in range(1, max_first_lags):
        gram_second_difference = (df[f"gram_lag_{i}"] - df[f"gram_lag_{i + first_lags_step}"]) - (
                    df[f"gram_lag_{i + first_lags_step}"] - df[f"gram_lag_{i + 2 * first_lags_step}"])
        df[f"gram_second_difference_{i}"] = gram_second_difference

        z_gram_second_difference = (df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + first_lags_step}"]) - (
                    df[f"z_gram_lag_{i + first_lags_step}"] - df[f"z_gram_lag_{i + 2 * first_lags_step}"])
        df[f"z_gram_second_difference_{i}"] = z_gram_second_difference

    for i in range(20, max_second_lags, second_lags_step):
        gram_second_difference = (df[f"gram_lag_{i}"] - df[f"gram_lag_{i + second_lags_step}"]) - (
                df[f"gram_lag_{i + second_lags_step}"] - df[f"gram_lag_{i + 2 * second_lags_step}"])
        df[f"gram_second_difference_{i}"] = gram_second_difference

        z_gram_second_difference = (df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + second_lags_step}"]) - (
                df[f"z_gram_lag_{i + second_lags_step}"] - df[f"z_gram_lag_{i + 2 * second_lags_step}"])
        df[f"z_gram_second_difference_{i}"] = z_gram_second_difference


def _append_distant_differences_feature(df: pd.DataFrame) -> None:
    factor = 10
    for i in range(1, 11):
        df[f"gram_distance_difference_{i}"] = df[f"gram_lag_{i}"] - df[f"gram_lag_{i * factor}"]
        df[f"z_gram_distance_difference_{i}"] = df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i * factor}"]


def _append_ratio_feature(df: pd.DataFrame) -> None:
    max_first_lags = 10
    first_lags_step = 1
    max_second_lags = 200
    second_lags_step = 10

    for i in range(1, max_first_lags):
        df[f"gram_ratio_{i}"] = df[f"gram_lag_{i}"] / df[f"gram_lag_{i + first_lags_step}"]
        df[f"z_gram_ratio_{i}"] = df[f"z_gram_lag_{i}"] / df[f"z_gram_lag_{i + first_lags_step}"]

    for i in range(20, max_second_lags, second_lags_step):
        df[f"gram_ratio_{i}"] = df[f"gram_lag_{i}"] / df[f"gram_lag_{i + second_lags_step}"]
        df[f"z_gram_ratio_{i}"] = df[f"z_gram_lag_{i}"] / df[f"z_gram_lag_{i + second_lags_step}"]


def _append_proportion_feature(df: pd.DataFrame) -> None:
    max_first_lags = 9
    first_lags_step = 1
    max_second_lags = 190
    second_lags_step = 10

    for i in range(1, max_first_lags):
        gram_proportion = (df[f"gram_lag_{i}"] - df[f"gram_lag_{i + first_lags_step}"]) / (
                df[f"gram_lag_{i + first_lags_step}"] - df[f"gram_lag_{i + 2 * first_lags_step}"])
        df[f"gram_proportion_{i}"] = gram_proportion

        z_gram_proportion = (df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + first_lags_step}"]) / (
                df[f"z_gram_lag_{i + first_lags_step}"] - df[f"z_gram_lag_{i + 2 * first_lags_step}"])
        df[f"z_gram_proportion_{i}"] = z_gram_proportion

    for i in range(20, max_second_lags, second_lags_step):
        gram_proportion = (df[f"gram_lag_{i}"] - df[f"gram_lag_{i + second_lags_step}"]) / (
                df[f"gram_lag_{i + second_lags_step}"] - df[f"gram_lag_{i + 2 * second_lags_step}"])
        df[f"gram_proportion_{i}"] = gram_proportion

        z_gram_proportion = (df[f"z_gram_lag_{i}"] - df[f"z_gram_lag_{i + second_lags_step}"]) / (
                df[f"z_gram_lag_{i + second_lags_step}"] - df[f"z_gram_lag_{i + 2 * second_lags_step}"])
        df[f"z_gram_proportion_{i}"] = z_gram_proportion


def _append_difference_between_distances_feature(df: pd.DataFrame) -> None:
    max_lag = 25
    for i in range(1, max_lag + 1):
        df[f"d_difference_lag_{i}"] = df["d"] - df[f"d_lag_{i}"]


def _append_non_linear_feature(df: pd.DataFrame) -> None:
    max_lag = 10
    for i in range(1, max_lag + 1):
        col = f"z_gram_lag_{i}"
        df[f"z_gram_sin_{i}"] = df[col].apply(lambda x: mp.sin(mp.mpf(x)) if pd.notnull(x) else np.nan)
        df[f"z_gram_sqrt_{i}"] = df[col].apply(lambda x: mp.sqrt(mp.mpf(x)) if pd.notnull(x) and x >= 0 else np.nan)


def generate_custom_dataset(limit: int = 100_000) -> None:
    print("Generating Custom dataset...")
    gram_points = dataset_loader_service.load_gram_points()
    distances = dataset_loader_service.load_distances()
    cogram_points = dataset_loader_service.load_cogram_points()
    zeros = dataset_loader_service.load_zeta_zeros()

    rows = []
    counter = 0

    for i, (gram, cogram, d, zero) in enumerate(zip(gram_points, cogram_points, distances, zeros)):
        row = {
            "gram": gram,
            "cogram": cogram,
            "d": d,
            "zero": zero,
            "z_gram": mp.siegelz(gram),
            "z_cogram": mp.siegelz(gram),
        }
        row.update(riemann_service.get_Z_function_terms_features(gram))
        row["z_integer"] = float(mp.siegelz(int(np.floor(gram))))
        rows.append(row)
        print(f"i: {i} | Gram: {gram} | Cogram: {cogram} | Distance: {d} | Zero: {zero}")

        counter += 1
        if counter >= limit:
            break

    df = pd.DataFrame(rows)
    df = _add_lags_to_custom_dataset(df)
    _append_first_difference_feature(df)
    _append_second_difference_feature(df)
    _append_distant_differences_feature(df)
    _append_ratio_feature(df)
    _append_proportion_feature(df)
    _append_difference_between_distances_feature(df)
    _append_non_linear_feature(df)
    df = df.drop(columns=["d"])
    print(df.head())

    df.to_csv("/app/dataset/custom.csv", index=False)
    print(f"[OK] Dataset gerado com shape: {df.shape}\n")
