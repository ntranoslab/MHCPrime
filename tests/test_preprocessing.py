import numpy as np
import pandas as pd
import pytest

from mhcprime.preprocessing import prepare_input_dataframe


def test_prepare_input_dataframe_accepts_minimal_input():
    df = pd.DataFrame({
        "seq": ["SLYNTVATL", "GILGFVFTL"],
        "allele": ["A0201", "A0201"],
    })

    out = prepare_input_dataframe(df)

    assert len(out) == 2
    assert "label" in out.columns
    assert "n_flank" in out.columns
    assert "c_flank" in out.columns
    assert "mhc_a_1" in out.columns
    assert "sa_ma" in out.columns

    assert out["label"].eq(0).all()
    assert out["seq"].astype(str).str.len().eq(14).all()
    assert out["n_flank"].astype(str).str.len().eq(10).all()
    assert out["c_flank"].astype(str).str.len().eq(10).all()


def test_unknown_allele_default_warns_and_drops():
    df = pd.DataFrame({
        "seq": ["SLYNTVATL"],
        "allele": ["Z9999"],
    })

    with pytest.warns(UserWarning):
        out = prepare_input_dataframe(df)

    assert len(out) == 0


def test_unknown_allele_strict_mode_raises():
    df = pd.DataFrame({
        "seq": ["SLYNTVATL"],
        "allele": ["Z9999"],
    })

    with pytest.raises(ValueError):
        prepare_input_dataframe(df, drop_missing_mhc=False)


def test_existing_missing_flanks_are_filled_safely():
    df = pd.DataFrame({
        "seq": ["SLYNTVATL"],
        "allele": ["A0201"],
        "n_flank": [np.nan],
        "c_flank": [""],
    })

    out = prepare_input_dataframe(df)

    assert len(out) == 1
    assert out["n_flank"].astype(str).str.len().iloc[0] == 10
    assert out["c_flank"].astype(str).str.len().iloc[0] == 10


def test_invalid_peptide_characters_raise():
    df = pd.DataFrame({
        "seq": ["SLYNTVAT!"],
        "allele": ["A0201"],
    })

    with pytest.raises(ValueError):
        prepare_input_dataframe(df)