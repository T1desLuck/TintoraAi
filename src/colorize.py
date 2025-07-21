import torch
import argparse
from PIL import Image
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image


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
    model = TintoraAI().to(device)
    try:
        model.load_state_dict(torch.load(
            "colorizer_weights.pth",
            map_location=device))
    except FileNotFoundError:
        print("Model weights not found. Output will be random. "
              "Train the model (see TRAINING.md).")
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
    print(f"Colored image saved to {args.output}")


if __name__ == "__main__":
    main()
