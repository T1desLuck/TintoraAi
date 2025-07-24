import torch
from src.model.tintora_ai import TintoraAI


def test_model_forward():
    model = TintoraAI(num_classes=1000)
    for size in [256, 512]:
        input_tensor = torch.randn(1, 1, size, size)
        try:
            color_output, semantic_output = model(input_tensor)
            assert color_output.shape == (1, 3, size, size), \
                   f"Color output shape mismatch for size {size}x{size}"
            assert semantic_output.shape == (1, 1000), \
                   f"Semantic output shape mismatch for size {size}x{size}"
            print(f"Color output shape for {size}x{size}: {color_output.shape}")
            print(f"Semantic output shape for {size}x{size}: {semantic_output.shape}")
        except Exception as e:
            print(f"Error testing size {size}x{size}: {e}")
            raise
