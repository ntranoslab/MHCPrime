import torch

from mhcprime.checkpointing import load_state_dict_flexible


def test_load_state_dict_flexible_loads_raw_state_dict(tmp_path):
    source = torch.nn.Linear(3, 1)
    target = torch.nn.Linear(3, 1)

    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(0.75)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint_path)

    load_state_dict_flexible(
        target,
        checkpoint_path,
        strict=True,
        map_location="cpu",
    )

    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])