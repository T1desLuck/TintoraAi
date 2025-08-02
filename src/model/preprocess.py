from PIL import Image
import torch
import numpy as np


def preprocess_image(image, pad_divisor=16, min_size=64):
    """
    Конвертирует изображение в ч/б и добавляет padding для совместимости с U-Net и GAN.

    Args:
        image: PIL Image
        pad_divisor: Делитель для padding (чтобы размеры были кратны делителю)
        min_size: Минимальный размер изображения

    Returns:
        tuple: (тензор изображения, исходный размер)
    """
    try:
        # Если изображение не в оттенках серого, конвертируем
        if image.mode != "L":
            image = image.convert("L")  # Чёрно-белое
    except Exception as e:
        raise ValueError(f"Ошибка при конвертации изображения: {e}")

    # Сохраняем исходный размер
    original_size = image.size
    w, h = original_size

    # Проверяем минимальный размер
    if w < min_size or h < min_size:
        # Вместо ошибки делаем resize до минимального размера
        larger_dimension = max(w, h)
        scale_factor = min_size / larger_dimension
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h
        print(f"Изображение увеличено до {w}x{h} пикселей")

    # Вычисляем padding для соответствия делителю (важно для U-Net)
    pad_w = (pad_divisor - w % pad_divisor) % pad_divisor
    pad_h = (pad_divisor - h % pad_divisor) % pad_divisor

    # Создаем новое изображение с padding
    padded_image = Image.new('L', (w + pad_w, h + pad_h), 0)
    padded_image.paste(image, (0, 0))

    # Нормализация и конвертация в тензор
    img_array = np.array(padded_image) / 255.0
    img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)

    print(f"Preprocessed tensor size: {img_tensor.shape}")
    return img_tensor, original_size
