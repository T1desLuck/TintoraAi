import torch
import argparse
from PIL import Image
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image


def main():
    parser = argparse.ArgumentParser(description="TintoraAI: Раскраска фотографий")
    parser.add_argument(
        "--input", type=str, required=True, help="Путь к входному фото")
    parser.add_argument(
        "--output", type=str, default="colored_image.jpg",
        help="Путь для сохранения результата")
    parser.add_argument(
        "--saturation", type=float, default=1.0,
        help="Насыщенность (0.5–2.0)")
    parser.add_argument(
        "--style", type=str, default="neutral",
        choices=["modern", "vintage", "neutral"], help="Стиль раскраски")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = TintoraAI().to(device)
    try:
        model.load_state_dict(torch.load(
            "colorizer_weights.pth",
            map_location=device))
    except FileNotFoundError:
        print("Веса модели не найдены. Результат будет случайным. "
              "Обучите модель (см. TRAINING.md).")
    model.eval()

    # Загрузка и обработка изображения
    image = Image.open(args.input)
    input_tensor, original_size = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Применение стиля
    if args.style == "modern":
        output_tensor *= 1.1  # Увеличение яркости
    elif args.style == "vintage":
        output_tensor *= 0.9  # Приглушенные тона

    # Постобработка
    colored_image = postprocess_image(output_tensor, original_size,
                                      args.saturation)
    colored_image.save(args.output)
    print(f"Раскрашенное фото сохранено в {args.output}")


if __name__ == "__main__":
    main()
