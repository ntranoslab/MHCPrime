import pandas as pd
import torch

from mhcprime import AminoAcidTokenizer, predict_dataframe_slow


class DummyMHCPrimeModel(torch.nn.Module):
    fc_output_dim = 1

    def forward(self, batch, return_embeddings=False):
        n = batch["peptide"].shape[0]
        logits = torch.arange(n, dtype=torch.float32).view(-1, 1)

        if return_embeddings:
            embeddings = torch.zeros((n, 4), dtype=torch.float32)
            return logits, embeddings

        return logits


def test_user_facing_output_drops_generated_flanks_and_unnamed_index():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "seq": ["SLYNTVATL", "GILGFVFTL"],
        "allele": ["A0201", "A0201"],
        "label": [1, 0],
    })

    out = predict_dataframe_slow(
        model=DummyMHCPrimeModel(),
        df=df,
        tokenizer=AminoAcidTokenizer(),
        score_col="mhcprime",
        batch_size=2,
        num_workers=0,
        device="cpu",
        add_rank=False,
        disable_tqdm=True,
    )

    assert list(out.columns) == ["seq", "allele", "label", "mhcprime"]
    assert "n_flank" not in out.columns
    assert "c_flank" not in out.columns
    assert "Unnamed: 0" not in out.columns
    assert len(out) == 2


def test_user_provided_flanks_are_preserved_in_user_facing_output():
    df = pd.DataFrame({
        "seq": ["SLYNTVATL", "GILGFVFTL"],
        "allele": ["A0201", "A0201"],
        "label": [1, 0],
        "n_flank": ["AA", ""],
        "c_flank": ["RR", ""],
    })

    out = predict_dataframe_slow(
        model=DummyMHCPrimeModel(),
        df=df,
        tokenizer=AminoAcidTokenizer(),
        score_col="mhcprime",
        batch_size=2,
        num_workers=0,
        device="cpu",
        add_rank=False,
        disable_tqdm=True,
    )

    assert "n_flank" in out.columns
    assert "c_flank" in out.columns
    assert out.loc[0, "n_flank"] == "AA"
    assert out.loc[0, "c_flank"] == "RR"
    assert "mhcprime" in out.columns