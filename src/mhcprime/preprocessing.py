import re
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd
import warnings

from .data.mhc_pseudo_seqs import mhc_seq_dict as RAW_MHC_SEQ_DICT
DEFAULT_ALLOWED_PEPTIDE_AA = set("ACDEFGHIKLMNPQRSTVWYUX")

def load_default_mhc_pseudosequences() -> Dict[str, str]:
    """
    Load the packaged MHC pseudosequence dictionary and normalize only the
    dictionary keys.

    User-provided dataframe alleles are expected to already match the public
    MHCPrime format, e.g. A0201, B0801, C0301, or any custom allele key that is
    present in the packaged/overridden pseudosequence dictionary.

    This function preserves the historical project behavior:
        HLA-A*02:01 -> A0201
        A*02:01     -> A0201
        A0201       -> A0201
    for dictionary keys only.
    """
    return {
        str(k).removeprefix("HLA-").replace("*", "").replace(":", ""): v
        for k, v in RAW_MHC_SEQ_DICT.items()
    }

def _strip_padding(seq: str, pad_char: str = "$") -> str:
    return str(seq).replace(pad_char, "")

def _validate_required_columns(
    df: pd.DataFrame,
    *,
    seq_col: str,
    allele_col: str,
) -> None:
    missing = [c for c in (seq_col, allele_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataframe is missing required column(s): {missing}. "
            f"Expected at minimum columns '{seq_col}' and '{allele_col}'."
        )

def _validate_peptides(
    df: pd.DataFrame,
    *,
    seq_col: str = "seq",
    min_len: int = 8,
    max_len: int = 14,
    allowed_aa: Iterable[str] = DEFAULT_ALLOWED_PEPTIDE_AA,
    pad_char: str = "$",
) -> None:
    """
    Validate peptide sequences before model padding.

    Existing '$' padding is ignored for length/AA validation, so users can pass
    either raw peptides or already right-padded peptides.
    """
    allowed_aa = set(allowed_aa)

    seq_unpadded = df[seq_col].astype(str).map(lambda x: _strip_padding(x, pad_char))
    lengths = seq_unpadded.str.len()

    bad_len_mask = (lengths < min_len) | (lengths > max_len)
    if bad_len_mask.any():
        bad_examples = (
            df.loc[bad_len_mask, seq_col]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Peptide lengths must be between {min_len} and {max_len} residues "
            f"before padding. Found {int(bad_len_mask.sum())} invalid peptide(s). "
            f"Examples: {bad_examples}"
        )

    bad_aa_mask = ~seq_unpadded.map(lambda s: set(s).issubset(allowed_aa))
    if bad_aa_mask.any():
        bad_examples = (
            df.loc[bad_aa_mask, seq_col]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Peptide sequences contain unsupported amino-acid characters. "
            f"Allowed amino acids: {''.join(sorted(allowed_aa))}. "
            f"Found {int(bad_aa_mask.sum())} invalid peptide(s). "
            f"Examples: {bad_examples}"
        )


def _validate_or_create_flanks(
    df: pd.DataFrame,
    *,
    n_flank_col: str = "n_flank",
    c_flank_col: str = "c_flank",
    flank_len: int = 10,
    pad_char: str = "$",
) -> pd.DataFrame:
    """
    Ensure n_flank and c_flank exist and are no longer than flank_len
    before padding.

    Missing flanks are filled with '$' * flank_len, matching your existing
    no-flank inference/training behavior.
    """
    df = df.copy()

    if n_flank_col not in df.columns:
        df[n_flank_col] = pad_char * flank_len

    if c_flank_col not in df.columns:
        df[c_flank_col] = pad_char * flank_len

    for col in (n_flank_col, c_flank_col):
        df[col] = df[col].fillna("").astype(str)

    for col in (n_flank_col, c_flank_col):
        unpadded = df[col].astype(str).map(lambda x: _strip_padding(x, pad_char))
        too_long = unpadded.str.len() > flank_len

        if too_long.any():
            examples = df.loc[too_long, col].astype(str).head(10).tolist()
            raise ValueError(
                f"Column '{col}' contains flank sequence(s) longer than "
                f"{flank_len} residues before padding. MHCPrime supports up to "
                f"{flank_len} N-terminal and {flank_len} C-terminal flank residues. "
                f"Found {int(too_long.sum())} invalid row(s). Examples: {examples}"
            )

    return df

def pad_sequence_column(
    df: pd.DataFrame,
    *,
    column: str,
    max_length: int,
    pad_char: str = "$",
    pad_side: str = "right",
) -> pd.DataFrame:
    """
    Pad a sequence column to a fixed length.

    This intentionally mirrors your existing right-padding behavior and does
    not truncate overlong sequences.
    """
    df = df.copy()

    if pad_side not in {"right", "left"}:
        raise ValueError("pad_side must be either 'right' or 'left'.")

    values = df[column].astype(str)

    if pad_side == "right":
        df[column] = values.map(lambda x: x.ljust(max_length, pad_char))
    else:
        df[column] = values.map(lambda x: x.rjust(max_length, pad_char))

    return df


def _mhc_column_for_allele(allele: str) -> str:
    """
    Return the single-allele MHC column for an allele.

    Historical behavior:
      A* alleles -> mhc_a_1
      B* alleles -> mhc_b_1
      C* alleles -> mhc_c_1

    New extension:
      Any non-A/B/C allele key -> mhc_a_1

    This preserves all current A/B/C behavior and only adds a fallback for
    custom or non-human allele names.
    """
    allele = str(allele)
    if len(allele) == 0:
        return "mhc_a_1"

    locus = allele[0].upper()
    if locus not in {"A", "B", "C"}:
        locus = "A"

    return f"mhc_{locus.lower()}_1"


def map_allele_pseudosequences(
    df: pd.DataFrame,
    mhc_seq_dict: Dict[str, str],
    *,
    allele_col: str = "allele",
    drop_missing: bool = True,
    warn_missing: bool = True,
) -> pd.DataFrame:
    """
    Map allele names to MHC pseudosequences.

    Rows whose allele is absent from mhc_seq_dict are removed by default, with
    a user-facing warning. To use such alleles, users must add the allele key
    and 34-residue pseudosequence to the dictionary.
    """
    df = df.copy()

    missing_mask = ~df[allele_col].astype(str).isin(mhc_seq_dict)
    if missing_mask.any():
        missing_alleles = sorted(df.loc[missing_mask, allele_col].astype(str).unique())
        n_missing_rows = int(missing_mask.sum())

        msg = (
            f"{n_missing_rows} row(s) contain allele(s) without packaged MHC "
            f"pseudosequences and will be removed: {missing_alleles}. "
            f"To score these rows, add the allele key and its 34-residue "
            f"pseudosequence to the MHC pseudosequence dictionary."
        )

        if warn_missing:
            warnings.warn(msg, UserWarning)

        if drop_missing:
            df = df.loc[~missing_mask].copy()
        else:
            raise ValueError(msg)

    # initialize all expected columns
    df["mhc_a_1"] = None
    df["mhc_b_1"] = None
    df["mhc_c_1"] = None

    # route each allele to A/B/C columns, with non-A/B/C fallback to mhc_a_1
    for idx, allele in df[allele_col].astype(str).items():
        mhc_col = _mhc_column_for_allele(allele)
        df.at[idx, mhc_col] = mhc_seq_dict[allele]

    return df


def prepare_input_dataframe(
    df: pd.DataFrame,
    *,
    mhc_seq_dict: Optional[Dict[str, str]] = None,
    seq_col: str = "seq",
    allele_col: str = "allele",
    label_col: str = "label",
    n_flank_col: str = "n_flank",
    c_flank_col: str = "c_flank",
    peptide_len: int = 14,
    flank_len: int = 10,
    min_peptide_len: int = 8,
    max_peptide_len: int = 14,
    pad_char: str = "$",
    remove_flank: bool = False,
    drop_missing_mhc: bool = True,
    warn_missing_mhc: bool = True,
    allowed_peptide_aa: Iterable[str] = DEFAULT_ALLOWED_PEPTIDE_AA,
) -> pd.DataFrame:
    """
    Prepare a user dataframe for MHCPrime inference or general supervised
    training.

    Expected user-provided columns:
        seq
        allele

    Optional columns:
        label
        n_flank
        c_flank

    Public assumption:
        User alleles are already in MHCPrime key format, e.g. A0201, B0801,
        C0301, or another custom key present in mhc_seq_dict.

    Output columns include the fields expected by PeptideMHCDataset:
        seq, allele, label, n_flank, c_flank,
        mhc_a_1, mhc_b_1, mhc_c_1,
        mhc_a_2, mhc_b_2, mhc_c_2,
        sa_ma
    """
    _validate_required_columns(df, seq_col=seq_col, allele_col=allele_col)

    out = df.copy()

    # Keep the public/internal column names stable.
    if seq_col != "seq":
        out = out.rename(columns={seq_col: "seq"})
    if allele_col != "allele":
        out = out.rename(columns={allele_col: "allele"})
    if label_col != "label" and label_col in out.columns:
        out = out.rename(columns={label_col: "label"})
    if n_flank_col != "n_flank" and n_flank_col in out.columns:
        out = out.rename(columns={n_flank_col: "n_flank"})
    if c_flank_col != "c_flank" and c_flank_col in out.columns:
        out = out.rename(columns={c_flank_col: "c_flank"})

    # Required string columns.
    out["seq"] = out["seq"].astype(str)
    out["allele"] = out["allele"].astype(str)

    # If label is absent, create dummy labels for inference.
    if "label" not in out.columns:
        out["label"] = 0

    _validate_peptides(
        out,
        seq_col="seq",
        min_len=min_peptide_len,
        max_len=max_peptide_len,
        allowed_aa=allowed_peptide_aa,
        pad_char=pad_char,
    )

    out = _validate_or_create_flanks(
        out,
        n_flank_col="n_flank",
        c_flank_col="c_flank",
        flank_len=flank_len,
        pad_char=pad_char,
    )

    if remove_flank:
        out["n_flank"] = pad_char * flank_len
        out["c_flank"] = pad_char * flank_len

    # Match the historical preprocessing: right-pad peptide and flanks.
    out = pad_sequence_column(
        out,
        column="seq",
        max_length=peptide_len,
        pad_char=pad_char,
        pad_side="right",
    )
    out = pad_sequence_column(
        out,
        column="n_flank",
        max_length=flank_len,
        pad_char=pad_char,
        pad_side="right",
    )
    out = pad_sequence_column(
        out,
        column="c_flank",
        max_length=flank_len,
        pad_char=pad_char,
        pad_side="right",
    )

    if mhc_seq_dict is None:
        mhc_seq_dict = load_default_mhc_pseudosequences()

    out = map_allele_pseudosequences(
        out,
        mhc_seq_dict=mhc_seq_dict,
        allele_col="allele",
        drop_missing=drop_missing_mhc,
        warn_missing=warn_missing_mhc,
    )

    # Secondary MHC columns retained for compatibility with existing dataset.
    out["mhc_a_2"] = None
    out["mhc_b_2"] = None
    out["mhc_c_2"] = None

    # Current public path is single-allele.
    out["sa_ma"] = "SA"

    out = out.reset_index(drop=True)
    return out

def unprocess_output_dataframe(
    df: pd.DataFrame,
    *,
    pad_char: str = "$",
    remove_model_columns: bool = True,
    strip_padding_cols: tuple[str, ...] = ("seq", "n_flank", "c_flank"),
    extra_drop_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Convert an internally processed MHCPrime dataframe back to a cleaner
    user-facing dataframe.

    This removes MHCPrime-internal model columns such as mhc_a_1/mhc_b_1/mhc_c_1
    and sa_ma, while preserving user-facing columns such as seq, allele, label,
    flanks, metadata columns, and score/rank columns.

    Padding is stripped from seq/n_flank/c_flank by default.
    """
    out = df.copy()

    for col in strip_padding_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace(pad_char, "", regex=False)

    if remove_model_columns:
        model_cols = {
            "mhc_a_1",
            "mhc_b_1",
            "mhc_c_1",
            "mhc_a_2",
            "mhc_b_2",
            "mhc_c_2",
            "sa_ma",
        }
        drop_cols = [c for c in model_cols if c in out.columns]
        out = out.drop(columns=drop_cols)

    if extra_drop_cols:
        drop_cols = [c for c in extra_drop_cols if c in out.columns]
        out = out.drop(columns=drop_cols)

    return out

# def replace_label_column(df, base_label_col, new_label_col):
#     if base_label_col == new_label_col:
#         return df
#     df = df.drop(columns=[base_label_col])
#     df = df.rename(columns={new_label_col: base_label_col})
#     return df

def create_seq_allele_column(df):
    df["seq_allele"]=[f"{s}_{a}" for s,a in zip(df.seq, df.allele)]
    return df

def filter_canonical_sequences(df, column_name, amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    allowed_set = set(amino_acids)
    return df[df[column_name].apply(lambda seq: set(seq).issubset(allowed_set))]

def remove_flanks(df):
    df["n_flank"]="$"*10
    df["c_flank"]="$"*10
    return df

def filter_seq_len(df, min_length=8, max_length=14):
    df["seq_len"]=df.seq.str.replace("$","").apply(len)
    df_f=df.query(f"seq_len >= {min_length} and seq_len <= {max_length}")
    return df_f
    
def standardize_allele_format(df: pd.DataFrame, no_hla_prefix=False) -> pd.DataFrame:
    df = df.copy()
    
    def clean_allele(allele: str) -> Optional[str]:
        if pd.isna(allele) or allele == 'Neg':
            return allele
            
        if '_' in allele:
            parts = allele.split('_')
            cleaned = [clean_allele(part) for part in parts]
            return '_'.join(sorted(filter(None, cleaned)))
            
        if re.match(r'^[A-C]\d{4}$', allele):
            return allele
            
        match = re.match(r'HLA-([A-C])\*(\d{2}):(\d{2})', allele)
        if no_hla_prefix:
            match = re.match(r'([A-C])\*(\d{2}):(\d{2})', allele)
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}"
            
        return None
    
    df['allele'] = df['allele'].apply(clean_allele)
    return df.dropna(subset=['allele'])


def map_allele_sequences_efficient_alt(df, allele_dict):
    df = df.copy()
    
    df["mhc_a_1"] = None
    df["mhc_b_1"] = None
    df["mhc_c_1"] = None
    df["temp_seq"] = df["allele"].map(allele_dict)
    df["allele_type"] = df["allele"].str[0:1]

    for prefix in ['A', 'B', 'C']:
        column_name = f"mhc_{prefix.lower()}_1"
        mask = df["allele_type"] == prefix
        df.loc[mask, column_name] = df.loc[mask, "temp_seq"]
    
    df = df.drop(columns=["temp_seq", "allele_type"])
    
    return df

def pad_peptides(df: pd.DataFrame, column: str = 'seq', max_length: int = 14, pad_char: str = '$', pad_side: str = 'right') -> pd.DataFrame:
    if pad_side == 'right':
        df[column] = df[column].apply(lambda x: x.ljust(max_length, pad_char))
    elif pad_side == 'left':
        df[column] = df[column].apply(lambda x: x.rjust(max_length, pad_char))
    return df

# NOTE: check here if flanks contain nans
def full_process_data(df, mhc_seq_dict, remove_flank=False, add_pseudo_label=False):
    df=pad_peptides(df)

    if "n_flank" not in df.columns.tolist():
        df["n_flank"]="$"*10

    if "c_flank" not in df.columns.tolist():
        df["c_flank"]="$"*10

    if remove_flank:
        df=remove_flanks(df)
    
    df=pad_peptides(df, column='n_flank', max_length=10)
    df=pad_peptides(df, column='c_flank', max_length=10)
    df=map_allele_sequences_efficient_alt(df, mhc_seq_dict)
    df["mhc_a_2"]=None
    df["mhc_b_2"]=None
    df["mhc_c_2"]=None
    df["sa_ma"]="SA"

    if add_pseudo_label:
        df["label"] = 0
    return df
