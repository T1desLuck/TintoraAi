from PIL import Image
import numpy as np
import torch


def postprocess_image(output_tensor, original_size, saturation=1.0):
    """
    Конвертирует тензор в изображение, обрезает до исходного размера.
    
    Args:
        output_tensor: Выходной тензор модели
        original_size: Исходный размер изображения (ширина, высота)
        saturation: Коэффициент насыщенности цветов
        
    Returns:
        PIL.Image: Обработанное цветное изображение
    """
    if saturation <= 0:
        raise ValueError("Saturation must be positive")
        
    w, h = original_size
    
    # Если тензор на GPU, перемещаем его на CPU
    if output_tensor.is_cuda:
        output_tensor = output_tensor.cpu()
        
    # Преобразуем тензор в массив numpy
    output = output_tensor.squeeze().permute(1, 2, 0).numpy()
    
    # Обрезаем до исходного размера (если размеры отличаются)
    if output.shape[0] > h or output.shape[1] > w:
        output = output[:h, :w, :]
    
    # Применяем насыщенность
    output = output * saturation
    
    # Преобразование в диапазон 0-255 для изображения
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(output)


def apply_color_enhancement(image, temperature=0, contrast=0, brightness=0):
    """
    Улучшает цветовые характеристики изображения.
    
    Args:
        image: PIL Image
        temperature: Температура цвета (-100 до 100, холодный-теплый)
        contrast: Контрастность (-100 до 100)
        brightness: Яркость (-100 до 100)
        
    Returns:
        PIL.Image: Обработанное изображение
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Применяем температуру (тёплые/холодные цвета)
    if temperature != 0:
        factor = temperature / 100
        if factor > 0:  # Теплее
            img_array[:, :, 0] += factor * 0.2  # Больше красного
            img_array[:, :, 2] -= factor * 0.1  # Меньше синего
        else:  # Холоднее
            factor = abs(factor)
            img_array[:, :, 0] -= factor * 0.1  # Меньше красного
            img_array[:, :, 2] += factor * 0.2  # Больше синего
    
    # Применяем контрастность
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img_array = factor * (img_array - 0.5) + 0.5
    
    # Применяем яркость
    if brightness != 0:
        img_array += brightness / 100
        
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)
