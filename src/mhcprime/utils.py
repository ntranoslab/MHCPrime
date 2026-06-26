import gc
import torch
import pickle
import pandas as pd

def save_dict_pickle(dictionary, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def clear_all_gpu_memory():

    torch.cuda.empty_cache()
    for obj in list(globals().values()):
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            del obj

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        except Exception as e:
            print(f"Error while deleting tensor: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("All GPU memory has been cleared.")
    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def get_trainable_parameters(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]

def replace_label_column(df, base_label_col, new_label_col):
    """
    Return a dataframe where new_label_col is renamed to base_label_col.

    If the columns are already the same, the dataframe is returned unchanged.
    """
    if base_label_col == new_label_col:
        return df

    df = df.copy()

    if new_label_col not in df.columns:
        raise ValueError(f"new_label_col='{new_label_col}' not found in dataframe.")

    if base_label_col in df.columns:
        df = df.drop(columns=[base_label_col])

    df = df.rename(columns={new_label_col: base_label_col})
    return df


def print_data_stats(df, allele_dist=False, label_col="label"):
    """
    Print simple peptide-MHC dataframe statistics.

    Expected columns:
        allele
        label_col

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    allele_dist : bool
        If True, print per-allele positive/negative counts.
    label_col : str
        Name of binary label column. This column is temporarily treated as
        'label' for summary printing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        print("Total: 0")
        print("Pos: 0 | 0.00%")
        print("Neg: 0 | 0.00%")
        print("N alleles: 0")
        return

    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in dataframe.")

    if "allele" not in df.columns:
        raise ValueError("Column 'allele' not found in dataframe.")

    d = replace_label_column(df, "label", label_col)

    total = d.shape[0]
    pos = d.query("label == 1").shape[0]
    neg = d.query("label == 0").shape[0]

    print(f"Total: {total:,}")
    print(f"Pos: {pos:,} | {(pos / total) * 100:.2f}%")
    print(f"Neg: {neg:,} | {(neg / total) * 100:.2f}%")
    print("N alleles:", d["allele"].nunique())

    if allele_dist:
        for allele in d["allele"].dropna().unique():
            df_a = d[d["allele"] == allele]
            total_a = df_a.shape[0]
            pos_a = df_a.query("label == 1").shape[0]
            neg_a = df_a.query("label == 0").shape[0]

            print(allele)
            print(f"  - Total: {total_a:,}")
            print(f"  - Pos: {pos_a:,} | {(pos_a / total_a) * 100:.2f}%")
            print(f"  - Neg: {neg_a:,} | {(neg_a / total_a) * 100:.2f}%")