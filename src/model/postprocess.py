from PIL import Image
import numpy as np


def postprocess_image(output_tensor, original_size, saturation=1.0):
    """Конвертирует тензор в изображение, обрезает padding,
    применяет насыщенность."""
    output = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output = output * saturation  # Настройка насыщенности
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    output = output[:original_size[1], :original_size[0]]  # Обрезка
    return Image.fromarray(output).resize(original_size, Image.LANCZOS)
