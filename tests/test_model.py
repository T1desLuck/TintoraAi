import torch
from src.model.tintora_ai import TintoraAI


def test_model_forward():
    model = TintoraAI()
    input_tensor = torch.randn(1, 1, 256, 256)
    output = model(input_tensor)
    assert output.shape == (1, 3, 256, 256), "Incorrect output shape"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output not in [0, 1]"
