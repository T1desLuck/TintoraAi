from PIL import Image
import torch
import numpy as np


def preprocess_image(image, pad_divisor=8):
    """Конвертирует изображение в ч/б и добавляет padding
    для совместимости с U-Net.
    """
    image = image.convert("L")  # Чёрно-белое
    w, h = image.size
    pad_w = (pad_divisor - w % pad_divisor) % pad_divisor
    pad_h = (pad_divisor - h % pad_divisor) % pad_divisor
    padded_image = Image.new(
        'L',
        (w + pad_w, h + pad_h),
        0
    ) # Добавление padding
    padded_image.paste(image, (0, 0))
    img_array = np.array(padded_image) / 255.0
    img_tensor = (torch.from_numpy(img_array)
                  .float()
                  .unsqueeze(0)
                  .unsqueeze(0))
    return img_tensor, (w, h)
