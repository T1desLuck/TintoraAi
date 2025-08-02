# Использование TintoraAI

## Колоризация одного изображения

```bash
python src/colorize.py --input path/to/bw_image.jpg --output path/to/color_image.jpg
```

---

## Пакетная обработка

```bash
python src/colorize.py --input path/to/bw_images/ --output path/to/color_images/ --batch 4
```

---

## Параметры

| Параметр        | Описание                                    | По умолчанию |
|-----------------|---------------------------------------------|--------------|
| --input         | Путь к ч/б изображению или папке            | (обязателен) |
| --output        | Путь для сохранения результата              | colored_image.jpg |
| --model         | Весы модели                                 | models/colorizer_weights.pth |
| --batch         | Размер батча                                | 1            |
| --saturation    | Насыщенность (0.5-2.0)                      | 1.0          |
| --style         | Стиль: modern, vintage, sepia, dramatic     | neutral      |
| --temperature   | Температура цвета (-100 до 100)             | 0            |
| --contrast      | Контрастность (-100 до 100)                 | 0            |
| --brightness    | Яркость (-100 до 100)                       | 0            |

---

## Примеры

- Современная колоризация:
  ```bash
  python src/colorize.py --input bw.jpg --output modern.jpg --style modern --saturation 1.2
  ```
- Винтажный стиль:
  ```bash
  python src/colorize.py --input bw.jpg --output vintage.jpg --style vintage
  ```
- Пакетная обработка:
  ```bash
  python src/colorize.py --input bw_folder/ --output color_folder/ --batch 8
  ```

---

## Google Colab

```python
!python src/colorize.py --input /content/input.jpg --output /content/output.jpg
```

---

## Vast.ai

```python
import subprocess
subprocess.run(["/workspace/venv/bin/python", "/workspace/TintoraAI/src/colorize.py", "--input", "/workspace/input.jpg", "--output", "/workspace/output.jpg"])
```

---

## Интеграция в Python

```python
import torch
from PIL import Image
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image

model = TintoraAI(num_classes=100)
model.load_state_dict(torch.load("models/colorizer_weights.pth"))
model.eval()
image = Image.open("bw.jpg")
input_tensor, orig_size = preprocess_image(image)
with torch.no_grad():
    color_output, _ = model(input_tensor)
result = postprocess_image(color_output, orig_size, saturation=1.2)
result.save("result.jpg")
```