import torch
import argparse
from PIL import Image
import numpy as np
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image


def apply_color_filter(image, style):
    """Применяет цветовой фильтр для заданного стиля."""
    img_array = np.array(image).astype(np.float32) / 255.0
    if style == "modern":
        # Увеличение яркости и контрастности
        img_array[:, :, 0] *= 1.1  # Усиление красного
        img_array[:, :, 1] *= 1.05  # Лёгкое усиление зелёного
        img_array[:, :, 2] *= 1.15  # Усиление синего
    elif style == "vintage":
        # Приглушённые тона с сепией
        img_array = img_array * np.array([1.0, 0.95, 0.9])  # Сепия-эффект
        img_array += np.array([0.05, 0.03, 0.0])  # Лёгкий коричневый оттенок
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def main():
    parser = argparse.ArgumentParser(
        description="TintoraAI: Colorization of photos"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default="colored_image.jpg",
        help="Path to save output image")
    parser.add_argument(
        "--saturation",
        type=float,
        default=1.0,
        help="Color saturation (0.5-2.0)")
    parser.add_argument(
        "--style",
        type=str,
        default="neutral",
        choices=["modern", "vintage", "neutral"],
        help="Colorization style")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = TintoraAI(num_classes=1000).to(device)
    try:
        model.load_state_dict(torch.load(
            "colorizer_weights.pth",
            map_location=device))
    except FileNotFoundError:
        print("Model weights not found. Output will be random. "
              "Train the model (see TRAINING.md).")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()

    # Загрузка и обработка изображения
    try:
        image = Image.open(args.input)
    except Exception as e:
        print(f"Error loading input image: {e}")
        return
    input_tensor, original_size = preprocess_image(image, min_size=256)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        color_output, _ = model(input_tensor)  # Распаковываем только цвет

    # Постобработка
    colored_image = postprocess_image(color_output, original_size, args.saturation)

    # Применение стиля
    if args.style != "neutral":
        colored_image = apply_color_filter(colored_image, args.style)

    colored_image.save(args.output)
    print(f"Colored image saved to {args.output}")


if __name__ == "__main__":
    main()
