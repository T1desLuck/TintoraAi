import torch
import torch.nn as nn
from src.model.tintora_ai import ObjectClassifier


class LightweightUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(LightweightUNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2)
        self.out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        d1 = self.dec1(p1)
        out = self.out(d1)
        return torch.sigmoid(out)


class LightweightTintoraAI(nn.Module):
    def __init__(self, num_classes=1000):
        super(LightweightTintoraAI, self).__init__()
        self.unet = LightweightUNet()
        self.classifier = ObjectClassifier(num_classes)

    def forward(self, x):
        if x.shape[2] < 256 or x.shape[3] < 256:
            raise ValueError("Input image too small. Minimum size is 256x256")
        color_output = self.unet(x)
        semantic_output = self.classifier(x)
        return color_output, semantic_output


def test_model_forward():
    print("Starting test_model_forward")
    model = LightweightTintoraAI(num_classes=1000)
    size = 256
    print(f"Testing input size {size}x{size}")
    input_tensor = torch.randn(1, 1, size, size)
    try:
        color_output, semantic_output = model(input_tensor)
        assert color_output.shape == (1, 3, size, size), \
               f"Color output shape mismatch for size {size}x{size}"
        assert semantic_output.shape == (1, 1000), \
               f"Semantic output shape mismatch for size {size}x{size}"
        print(f"Color output shape: {color_output.shape}")
        print(f"Semantic output shape: {semantic_output.shape}")
    except Exception as e:
        print(f"Error testing size {size}x{size}: {e}")
        raise
    print("Test completed successfully")
