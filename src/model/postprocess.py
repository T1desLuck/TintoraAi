from PIL import Image
import numpy as np


def postprocess_image(output_tensor, original_size, saturation=1.0):
    """Конвертирует тензор в изображение, обрезает до исходного размера."""
    w, h = original_size
    output = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output = output[:h, :w, :]  # Обрезка до исходного размера
    output = output * saturation
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)
