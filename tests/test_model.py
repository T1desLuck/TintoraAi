import torch
from src.model.tintora_ai import TintoraAI


def test_model_forward():
    model = TintoraAI()
    input_tensor = torch.randn(1, 1, 256, 256)
    color_output, semantic_output = model(input_tensor)
    assert color_output.shape == (1, 3, 256, 256), (
        "Color output shape mismatch"
    )
    assert semantic_output.shape == (1, 10), "Semantic output shape mismatch"
