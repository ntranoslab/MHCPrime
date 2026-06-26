from importlib.resources import files
import pandas as pd

def load_example_dataset(
    name: str = "small",
    *,
    index_col=None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    name = str(name).lower().strip()
    dataset_files = {
        "small": "ms_test_data_small.csv.gz",
        "large": "ms_test_data_large.csv.gz",
    }

    if name not in dataset_files:
        raise ValueError(
            f"Unknown example dataset '{name}'. "
            f"Expected one of: {sorted(dataset_files)}"
        )

    path = files("mhcprime.data").joinpath(dataset_files[name])
    df = pd.read_csv(path, index_col=index_col, **read_csv_kwargs)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df