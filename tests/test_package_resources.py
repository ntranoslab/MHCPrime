from importlib.resources import files


def test_package_import_and_packaged_resources():
    import mhcprime

    assert hasattr(mhcprime, "__version__")

    from mhcprime import get_default_checkpoint_path, load_example_dataset

    checkpoint_path = get_default_checkpoint_path()
    assert checkpoint_path.is_file()

    data_dir = files("mhcprime.data")
    assert data_dir.joinpath("aa_property_table.txt").is_file()
    assert data_dir.joinpath("processed_feature_table.csv").is_file()
    assert data_dir.joinpath("mhcprime_global_background_scores.npz").is_file()

    df = load_example_dataset("small")
    assert len(df) > 0
    assert "seq" in df.columns
    assert "allele" in df.columns
    assert "label" in df.columns
    assert "Unnamed: 0" not in df.columns