from PIL import Image
import numpy as np
import torch

def preprocess_image(image, base_size=256):
    """Конвертирует изображение в ч/б, масштабирует и добавляет padding."""
    image = image.convert("L")  # Ч/б
    w, h = image.size
    scale = base_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Padding до кратного 16
    pad_w = (16 - new_w % 16) % 16
    pad_h = (16 - new_h % 16) % 16
    image = Image.fromarray(np.pad(np.array(image), ((0, pad_h), (0, pad_w)), mode='reflect'))
    
    img_array = np.array(image) / 255.0
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor, (w, h)
