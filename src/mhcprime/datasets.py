from importlib.resources import files
import pandas as pd

def load_example_dataset(
    name: str = "small",
    *,
    index_col=None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a packaged example MHCPrime dataframe.

    Parameters
    ----------
    name : {"small", "large"}
        Which packaged example dataset to load.
    index_col : int, str, or None
        Passed to pandas.read_csv. Default None because the example
        peptide-MHC files are expected to contain real columns like
        seq, allele, and optionally label.
    **read_csv_kwargs
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        Unprocessed dataframe with columns such as seq, allele, and label.
    """
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
    return pd.read_csv(path, index_col=index_col, **read_csv_kwargs)