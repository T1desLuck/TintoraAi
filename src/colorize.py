import torch
import argparse
import os
from PIL import Image
import numpy as np
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image, apply_color_enhancement


def apply_color_filter(image, style):
    """Применяет цветовой фильтр для заданного стиля."""
    img_array = np.array(image).astype(np.float32) / 255.0
    
    if style == "modern":
        # Современный яркий стиль
        img_array[:, :, 0] *= 1.1  # Увеличиваем красный
        img_array[:, :, 1] *= 1.05  # Немного увеличиваем зеленый
        img_array[:, :, 2] *= 1.15  # Больше синего для холодных тонов
        
    elif style == "vintage":
        # Винтажный стиль с теплыми тонами
        img_array = img_array * np.array([1.0, 0.95, 0.9])  # Уменьшаем зеленый и синий
        img_array += np.array([0.05, 0.03, 0.0])  # Добавляем красный и немного зеленого
        
    elif style == "sepia":
        # Сепия
        r, g, b = 0.393, 0.769, 0.189
        sepia_matrix = np.array([
            [r, g, b],
            [r * 0.9, g * 0.9, b * 0.9],
            [r * 0.7, g * 0.7, b * 0.7]
        ])
        
        flat_img = img_array.reshape(-1, 3)
        flat_img = flat_img @ sepia_matrix.T
        img_array = flat_img.reshape(img_array.shape)
        
    elif style == "dramatic":
        # Драматический контрастный стиль
        img_array = np.power(img_array, 0.9)  # Повышаем контраст
        img_array[:, :, 2] *= 1.2  # Больше синего для драматического эффекта
        
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def batch_colorize(input_dir, output_dir, model_path, batch_size=4, saturation=1.0,
                   style="neutral", temperature=0, contrast=0, brightness=0):
    """Обработка нескольких изображений в пакетном режиме"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели
    model = TintoraAI(num_classes=100).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False
        
    model.eval()
    
    # Создаем директорию для выходных файлов, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим все изображения в директории
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"В директории {input_dir} не найдены изображения")
        return False
        
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Обрабатываем изображения пакетами
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_inputs = []
        original_sizes = []
        
        # Подготавливаем каждое изображение
        for filename in batch_files:
            try:
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path)
                input_tensor, orig_size = preprocess_image(image)
                batch_inputs.append(input_tensor)
                original_sizes.append((orig_size, filename))
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")
                continue
        
        if not batch_inputs:
            continue
            
        # Объединяем в один батч
        batch_tensor = torch.cat(batch_inputs, dim=0).to(device)
        
        # Выполняем колоризацию
        with torch.no_grad():
            color_outputs, _ = model(batch_tensor)
        
        # Обрабатываем каждое изображение в батче
        for j, output_tensor in enumerate(color_outputs):
            if j >= len(original_sizes):
                break
                
            orig_size, filename = original_sizes[j]
            
            # Постобработка
            colored_image = postprocess_image(output_tensor.unsqueeze(0), orig_size, saturation)
            
            # Применение стиля
            if style != "neutral":
                colored_image = apply_color_filter(colored_image, style)
                
            # Применение дополнительных улучшений
            if temperature != 0 or contrast != 0 or brightness != 0:
                colored_image = apply_color_enhancement(
                    colored_image, temperature, contrast, brightness
                )
            
            # Сохраняем результат
            output_path = os.path.join(output_dir, f"colored_{filename}")
            colored_image.save(output_path)
            print(f"Сохранено: {output_path}")
    
    print("Обработка завершена")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TintoraAI: Нейронная сеть для колоризации черно-белых "
                   "или выцветших фотографий"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Путь к входному изображению или директории с изображениями")
    parser.add_argument("--output", type=str, default="colored_image.jpg",
                        help="Путь для сохранения выходного изображения или директории")
    parser.add_argument("--model", type=str, default="models/colorizer_weights.pth",
                        help="Путь к весам модели")
    parser.add_argument("--batch", type=int, default=1,
                        help="Размер батча для обработки нескольких изображений")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Насыщенность цвета (0.5-2.0)")
    parser.add_argument("--style", type=str, default="neutral",
                        choices=["modern", "vintage", "sepia", "dramatic", "neutral"],
                        help="Стиль колоризации")
    parser.add_argument("--temperature", type=int, default=0,
                        help="Температура цвета (-100 до 100, холодный-теплый)")
    parser.add_argument("--contrast", type=int, default=0,
                        help="Контрастность (-100 до 100)")
    parser.add_argument("--brightness", type=int, default=0,
                        help="Яркость (-100 до 100)")

    args = parser.parse_args()

    # Проверяем, является ли вход директорией
    if os.path.isdir(args.input):
        if args.output == "colored_image.jpg":
            args.output = "colored_images"
            
        # Пакетная обработка директории
        batch_colorize(
            args.input, args.output, args.model, args.batch,
            args.saturation, args.style, args.temperature, args.contrast, args.brightness
        )
    else:
        # Обработка одного изображения
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")

        # Загрузка модели
        model = TintoraAI(num_classes=100).to(device)
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
        except FileNotFoundError:
            print(f"Веса модели не найдены по пути {args.model}. "
                  f"Выходные данные будут случайными. Обучите модель (см. README.md).")
        except Exception as e:
            print(f"Ошибка загрузки весов модели: {e}")
            return
            
        model.eval()

        # Загрузка и обработка изображения
        try:
            image = Image.open(args.input)
            print(f"Загружено изображение размером {image.size[0]}x{image.size[1]}")
        except Exception as e:
            print(f"Ошибка загрузки входного изображения: {e}")
            return
            
        input_tensor, original_size = preprocess_image(image)
        input_tensor = input_tensor.to(device)

        # Колоризация
        with torch.no_grad():
            print("Выполняется колоризация...")
            color_output, _ = model(input_tensor)

        # Постобработка
        colored_image = postprocess_image(color_output, original_size, args.saturation)

        # Применение стиля
        if args.style != "neutral":
            print(f"Применяется стиль {args.style}...")
            colored_image = apply_color_filter(colored_image, args.style)

        # Применение дополнительных улучшений
        if args.temperature != 0 or args.contrast != 0 or args.brightness != 0:
            print("Применяются дополнительные улучшения...")
            colored_image = apply_color_enhancement(
                colored_image, args.temperature, args.contrast, args.brightness
            )

        # Сохранение результата
        colored_image.save(args.output)
        print(f"Цветное изображение сохранено в {args.output}")


if __name__ == "__main__":
    main()
