from importlib.resources import files
from pathlib import Path
from typing import Literal, Optional, Sequence, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

RankMode = Literal["global", "allele", "allele_len"]


def save_global_background_scores(
    background_df: pd.DataFrame,
    *,
    score_col: str,
    out_path: Union[str, Path],
    dtype: str = "float32",
) -> None:
    """
    Save sorted global background scores for lightweight percentile-rank
    computation.

    This preserves the rank behavior of add_background_percentile_ranks with
    rank_mode='global' and reverse=True/False, without storing the full
    background dataframe.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scores = pd.to_numeric(background_df[score_col], errors="coerce").to_numpy(
        dtype=np.float64,
        copy=False,
    )
    scores = scores[np.isfinite(scores)]
    scores = np.sort(scores).astype(dtype, copy=False)

    np.savez_compressed(
        out_path,
        scores=scores,
        score_col=np.array(score_col),
    )


def load_global_background_scores(
    path: Union[str, Path] = "mhcprime_global_background_scores.npz",
    *,
    package_data: bool = True,
) -> np.ndarray:
    """
    Load sorted global background scores.

    By default, this loads the packaged MHCPrime background score file:

        src/mhcprime/data/mhcprime_global_background_scores.npz

    Parameters
    ----------
    path:
        File name or path to the .npz background score file. If package_data=True,
        this is interpreted relative to mhcprime.data.
    package_data:
        If True, load from package resources. If False, load from a normal
        filesystem path.
    """
    if package_data:
        path = files("mhcprime.data").joinpath(str(path))
    else:
        path = Path(path)

    data = np.load(path)
    scores = data["scores"]

    if scores.size == 0:
        raise ValueError(f"No background scores found in {path}.")

    # Ensure searchsorted correctness even if user-created file was not sorted.
    if scores.size > 1 and np.any(scores[1:] < scores[:-1]):
        scores = np.sort(scores)

    return scores

def add_global_background_percentile_ranks(
    test_df: pd.DataFrame,
    *,
    background_scores: Union[np.ndarray, Sequence[float]],
    score_cols: Sequence[str],
    suffix: str = "_rank",
    reverse: bool = True,
    copy: bool = True,
    dtype: str = "float32",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add global background percentile-rank columns using pre-sorted or sortable
    background score values.

    This matches the global-mode behavior of add_background_percentile_ranks:

        reverse=True:
            rank = percent of background scores <= test score

        reverse=False:
            rank = percent of background scores >= test score

    For MHCPrime, reverse=True is the usual setting when larger model scores
    indicate stronger predicted presentation.
    """
    out_df = test_df.copy() if copy else test_df

    bg_sorted = np.asarray(background_scores, dtype=np.float64)
    bg_sorted = bg_sorted[np.isfinite(bg_sorted)]

    if bg_sorted.size == 0:
        raise ValueError("background_scores contains no finite values.")

    if bg_sorted.size > 1 and np.any(bg_sorted[1:] < bg_sorted[:-1]):
        bg_sorted = np.sort(bg_sorted)

    iterator = tqdm(
        score_cols,
        leave=True,
        disable=not show_progress,
        desc="Ranking score columns",
    )

    for score_col in iterator:
        if score_col not in out_df.columns:
            raise ValueError(f"score_col='{score_col}' not found in test_df.")

        test_vals = pd.to_numeric(out_df[score_col], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )

        out = np.full(len(out_df), np.nan, dtype=np.float64)
        valid = np.isfinite(test_vals)

        if valid.any():
            if reverse:
                pct = (
                    100.0
                    * np.searchsorted(bg_sorted, test_vals[valid], side="right")
                    / bg_sorted.size
                )
            else:
                pct = (
                    100.0
                    * (
                        bg_sorted.size
                        - np.searchsorted(bg_sorted, test_vals[valid], side="left")
                    )
                    / bg_sorted.size
                )

            out[valid] = pct

        out_df[f"{score_col}{suffix}"] = out.astype(dtype, copy=False)

    return out_df

RankMode = Literal["global", "allele", "allele_len"]

def add_background_percentile_ranks(
    test_df: pd.DataFrame,
    background_df: pd.DataFrame,
    allele_col: str = "allele",
    seq_len_col: str = "seq_len",
    score_cols: Optional[Sequence[str]] = None,
    rank_mode: RankMode = "global",
    suffix: str = "_rank",
    reverse: bool = True,
    copy: bool = True,
    dtype: str = "float32",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Add percentile-rank columns to `test_df` by placing each test peptide score into
    the score distribution of `background_df`.
    """
    if copy:
        out_df = test_df.copy()
    else:
        out_df = test_df

    if score_cols is None:
        excluded = {allele_col, seq_len_col}
        score_cols = [
            c for c in test_df.columns
            if c in background_df.columns and c not in excluded
        ]
    else:
        score_cols = [c for c in score_cols if c in test_df.columns and c in background_df.columns]

    if len(score_cols) == 0:
        raise ValueError("No valid score columns found in both test_df and background_df.")

    bg_gid, test_gid = _build_group_ids(
        background_df=background_df,
        test_df=test_df,
        allele_col=allele_col,
        seq_len_col=seq_len_col,
        rank_mode=rank_mode,
    )

    n_test = len(test_df)
    n_bg = len(background_df)

    bg_order, bg_unique_gids, bg_starts, bg_counts = _group_structure(bg_gid)
    test_order, test_unique_gids, test_starts, test_counts = _group_structure(test_gid)

    bg_gid_to_pos = {gid: i for i, gid in enumerate(bg_unique_gids)}

    iterator = tqdm(score_cols, leave=True, disable=not show_progress, desc="Ranking score columns")

    for score_col in iterator:
        bg_vals = pd.to_numeric(background_df[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        test_vals = pd.to_numeric(test_df[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        bg_vals_ord = bg_vals[bg_order]
        test_vals_ord = test_vals[test_order]

        out_ord = np.full(n_test, np.nan, dtype=np.float64)

        if rank_mode == "global":
            valid_bg = np.isfinite(bg_vals_ord)
            bg_sorted = np.sort(bg_vals_ord[valid_bg])

            valid_test = np.isfinite(test_vals_ord)
            if bg_sorted.size > 0 and valid_test.any():
                if reverse:
                    pct = 100.0 * np.searchsorted(bg_sorted, test_vals_ord[valid_test], side="right") / bg_sorted.size
                else:
                    pct = 100.0 * (bg_sorted.size - np.searchsorted(bg_sorted, test_vals_ord[valid_test], side="left")) / bg_sorted.size
                out_ord[valid_test] = pct

        else:
            for gid, t_start, t_count in zip(test_unique_gids, test_starts, test_counts):
                if gid < 0:
                    continue

                bg_pos = bg_gid_to_pos.get(gid, None)
                if bg_pos is None:
                    continue

                b_start = bg_starts[bg_pos]
                b_count = bg_counts[bg_pos]

                bg_slice = bg_vals_ord[b_start:b_start + b_count]
                test_slice = test_vals_ord[t_start:t_start + t_count]

                valid_bg = np.isfinite(bg_slice)
                if not valid_bg.any():
                    continue

                bg_sorted = np.sort(bg_slice[valid_bg])

                valid_test = np.isfinite(test_slice)
                if not valid_test.any():
                    continue

                if reverse:
                    pct = 100.0 * np.searchsorted(bg_sorted, test_slice[valid_test], side="right") / bg_sorted.size
                else:
                    pct = 100.0 * (bg_sorted.size - np.searchsorted(bg_sorted, test_slice[valid_test], side="left")) / bg_sorted.size

                tmp = np.full(t_count, np.nan, dtype=np.float64)
                tmp[valid_test] = pct
                out_ord[t_start:t_start + t_count] = tmp

        out = np.empty(n_test, dtype=np.float64)
        out[test_order] = out_ord

        out_df[f"{score_col}{suffix}"] = out.astype(dtype, copy=False)

    return out_df


def _group_structure(group_ids: np.ndarray):
    """
    Return sorted order and contiguous group metadata.
    """
    order = np.argsort(group_ids, kind="mergesort")
    gids_ord = group_ids[order]

    if gids_ord.size == 0:
        return order, np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    change = np.empty(gids_ord.size, dtype=bool)
    change[0] = True
    change[1:] = gids_ord[1:] != gids_ord[:-1]

    starts = np.flatnonzero(change)
    unique_gids = gids_ord[starts]
    counts = np.diff(np.append(starts, gids_ord.size))

    return order, unique_gids, starts, counts


def _build_group_ids(
    background_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allele_col: str,
    seq_len_col: str,
    rank_mode: RankMode,
):
    """
    Build integer group IDs for background and test in a way that test groups absent
    from background become -1.
    """
    n_bg = len(background_df)
    n_test = len(test_df)

    if rank_mode == "global":
        bg_gid = np.zeros(n_bg, dtype=np.int64)
        test_gid = np.zeros(n_test, dtype=np.int64)
        return bg_gid, test_gid

    bg_alleles = pd.Index(pd.unique(background_df[allele_col]))
    bg_allele_codes = pd.Categorical(background_df[allele_col], categories=bg_alleles).codes.astype(np.int64, copy=False)
    test_allele_codes = pd.Categorical(test_df[allele_col], categories=bg_alleles).codes.astype(np.int64, copy=False)

    if rank_mode == "allele":
        return bg_allele_codes, test_allele_codes

    if rank_mode != "allele_len":
        raise ValueError("rank_mode must be one of {'global', 'allele', 'allele_len'}")

    bg_len = pd.to_numeric(background_df[seq_len_col], errors="coerce").to_numpy(np.int64, copy=False)
    test_len = pd.to_numeric(test_df[seq_len_col], errors="coerce").to_numpy(np.int64, copy=False)

    min_len = min(bg_len.min(initial=0), test_len.min(initial=0))
    if min_len < 0:
        bg_len = bg_len - min_len
        test_len = test_len - min_len

    max_len = int(max(bg_len.max(initial=0), test_len.max(initial=0)))
    stride = max_len + 1

    bg_comp = bg_allele_codes * stride + bg_len

    valid_test = test_allele_codes >= 0
    test_comp = np.full(n_test, -1, dtype=np.int64)
    test_comp[valid_test] = test_allele_codes[valid_test] * stride + test_len[valid_test]

    unique_bg_comp, bg_gid = np.unique(bg_comp, return_inverse=True)
    test_gid = np.full(n_test, -1, dtype=np.int64)

    valid_test2 = test_comp >= 0
    if valid_test2.any():
        mapped = np.searchsorted(unique_bg_comp, test_comp[valid_test2])
        in_range = mapped < unique_bg_comp.size
        found = np.zeros(valid_test2.sum(), dtype=bool)
        found[in_range] = unique_bg_comp[mapped[in_range]] == test_comp[valid_test2][in_range]
        test_gid_valid = np.full(valid_test2.sum(), -1, dtype=np.int64)
        test_gid_valid[found] = mapped[found]
        test_gid[valid_test2] = test_gid_valid

    return bg_gid, test_gid
