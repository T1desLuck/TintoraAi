import torch
from src.model.tintora_ai import TintoraAI


def test_model_forward():
    model = TintoraAI(num_classes=1000)
    input_tensor = torch.randn(1, 1, 64, 64)  # Уменьшен до 64x64
    color_output, semantic_output = model(input_tensor)
    assert color_output.shape == (1, 3, 64, 64), \
           "Color output shape mismatch"
    assert semantic_output.shape == (1, 1000), \
           "Semantic output shape mismatch"
    print(f"Color output shape: {color_output.shape}")
    print(f"Semantic output shape: {semantic_output.shape}")
