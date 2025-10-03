import pandas as pd


def get_lagged_dataframe(series: pd.Series, max_lag: int, prefix: str, start_lag_at: int = 1, step: int = 1) -> pd.DataFrame:
    df = pd.DataFrame()
    for k in range(start_lag_at, max_lag + 1, step):
        df[f"{prefix}_lag_{k}"] = series.shift(k)
    return df
