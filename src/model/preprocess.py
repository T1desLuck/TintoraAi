from PIL import Image
import torch
import numpy as np


def preprocess_image(image, pad_divisor=8, min_size=256):
    """Конвертирует изображение в ч/б и добавляет padding
    для совместимости с U-Net.
    """
    try:
        image = image.convert("L")  # Чёрно-белое
    except Exception as e:
        raise ValueError(f"Ошибка при конвертации изображения: {e}")

    w, h = image.size
    if w < min_size or h < min_size:
        raise ValueError(f"Изображение слишком маленькое. Минимальный размер {min_size}x{min_size} пикселей")

    pad_w = (pad_divisor - w % pad_divisor) % pad_divisor
    pad_h = (pad_divisor - h % pad_divisor) % pad_divisor
    padded_image = Image.new('L', (w + pad_w, h + pad_h), 0)
    padded_image.paste(image, (0, 0))
    img_array = np.array(padded_image) / 255.0
    img_tensor = (torch.from_numpy(img_array)
                  .float()
                  .unsqueeze(0)
                  .unsqueeze(0))
    print(f"Preprocessed tensor size: {img_tensor.shape}")
    return img_tensor, (w, h)
