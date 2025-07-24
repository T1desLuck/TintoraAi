# Обучение TintoraAI

Хочешь обучить **TintoraAI** на своих данных, чтобы раскрашивать чёрно-белые фото ещё лучше? Этот гайд объяснит всё шаг за шагом, даже если ты новичок. Мы расскажем, как подготовить датасет и запустить обучение, чтобы всё было просто и понятно.

## Что нужно для обучения
- Установленный TintoraAI (смотри [INSTALL.md](INSTALL.md)).
- Компьютер с процессором (CPU) или видеокартой (GPU NVIDIA с CUDA для ускорения).
- Датасет с чёрно-белыми и цветными фото + метками объектов.
- Терпение: обучение может занять время, особенно на CPU.

## Подготовка датасета
TintoraAI ожидает датасет с тремя папками: `bw` (чёрно-белые фото), `color` (цветные фото) и `labels` (метки объектов). Вот как это сделать:

### 1. Собери изображения
- Найди пары изображений: чёрно-белое и его цветная версия. Минимальный размер — 256x256 пикселей.
- Источники:
  - Личные фото (отсканируй старые ч/б фото и найди их цветные аналоги).
  - Открытые фотостоки: Unsplash, Pexels, OpenImages.
  - Наборы данных: ImageNet (требует обработки).
- Для тестов хватит 100–500 пар изображений. Для хорошего результата — 10,000+ пар.

### 2. Создай структуру папок
- Создай папку `dataset` в корне проекта (`TintoraAi/dataset`).
- Внутри сделай три подпапки:
  ```bash
  mkdir dataset
  mkdir dataset/bw
  mkdir dataset/color
  mkdir dataset/labels
  ```
- Положи файлы:
  - `dataset/bw/photo1.jpg` — чёрно-белое фото.
  - `dataset/color/photo1.jpg` — цветная версия того же фото.
  - `dataset/labels/photo1.npy` — файл с меткой (например, число 0 для "трава").

### 3. Подготовь чёрно-белые изображения
- Если у тебя есть цветные фото, но нет чёрно-белых, создай их с помощью Python:
  ```python
  from PIL import Image
  import os

  color_dir = "dataset/color"
  bw_dir = "dataset/bw"
  os.makedirs(bw_dir, exist_ok=True)

  for file in os.listdir(color_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          img = Image.open(os.path.join(color_dir, file)).convert("L")
          img.save(os.path.join(bw_dir, file))
  ```
- Сохрани этот код как `convert_to_bw.py` и запусти:
  ```bash
  python convert_to_bw.py
  ```
- Проверь, что все файлы в `bw/` и `color/` имеют одинаковые имена и размеры (≥256x256).

### 4. Создай метки (`.npy`)
- Метки — это числа от 0 до 999 (для 1000 классов), обозначающие объекты на фото (например, 0 — "трава", 1 — "небо"). Для тестов можно использовать 10–100 классов.
- Если у тебя нет меток, создай простые (например, 0 для всех изображений):
  ```python
  import numpy as np
  import os

  bw_dir = "dataset/bw"
  label_dir = "dataset/labels"
  os.makedirs(label_dir, exist_ok=True)

  for file in os.listdir(bw_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          label = np.array([0])  # Пример: класс 0 для всех
          np.save(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")), label)
  ```
- Сохрани как `create_labels.py` и запусти:
  ```bash
  python create_labels.py
  ```
- Для реальных меток используй классификатор (например, ResNet от `torchvision`):
  ```python
  from torchvision import models, transforms
  import torch
  from PIL import Image
  import numpy as np
  import os

  model = models.resnet50(weights='IMAGENET1K_V1').eval()
  preprocess = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  color_dir = "dataset/color"
  label_dir = "dataset/labels"
  os.makedirs(label_dir, exist_ok=True)

  for file in os.listdir(color_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          img = Image.open(os.path.join(color_dir, file)).convert("RGB")
          img_tensor = preprocess(img).unsqueeze(0)
          with torch.no_grad():
              output = model(img_tensor)
              label = output.argmax().item()  # Класс с наивысшей вероятностью
          np.save(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")), np.array([label]))
  ```
- Сохрани как `generate_labels.py` и запусти:
  ```bash
  python generate_labels.py
  ```

### 5. Проверь датасет
- Убедись, что:
  - Все файлы в `bw/` и `color/` имеют одинаковые имена и размеры (≥256x256).
  - В `labels/` есть `.npy` файлы для каждого изображения.
  - Метки — числа от 0 до 999 (или меньше, если изменил `num_classes`).
- Пример проверки:
  ```python
  from PIL import Image
  import numpy as np
  import os

  bw_dir = "dataset/bw"
  color_dir = "dataset/color"
  label_dir = "dataset/labels"

  for file in os.listdir(bw_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          bw_img = Image.open(os.path.join(bw_dir, file))
          color_img = Image.open(os.path.join(color_dir, file))
          label = np.load(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")))
          assert bw_img.size == color_img.size, f"Size mismatch for {file}"
          assert bw_img.size[0] >= 256 and bw_img.size[1] >= 256, f"Image {file} too small"
          assert 0 <= label[0] < 1000, f"Invalid label for {file}"
  ```
- Сохрани как `check_dataset.py` и запусти:
  ```bash
  python check_dataset.py
  ```

## Запуск обучения

### 1. Перейди в папку проекта
```bash
cd TintoraAi
```

### 2. Запусти обучение
- Базовая команда:
  ```bash
  python src/training/train.py --data_path dataset
  ```
- Настраиваемые параметры:
  - `--epochs 10`: Количество эпох (10 для теста, 50+ для хорошего результата).
  - `--batch_size 4`: Размер батча (4 для слабых GPU/CPU, 8–16 для мощных).
  - `--num_classes 100`: Количество классов для семантического анализа (100 для теста, 1000 по умолчанию).
  Пример:
  ```bash
  python src/training/train.py --data_path dataset --epochs 10 --batch_size 4 --num_classes 100
  ```

### 3. Следи за процессом
- В терминале будут строки вроде:
  ```
  Epoch 1/10, Loss: 0.85
  ```
- Меньший `Loss` означает, что модель учится лучше.
- Если обучение тормозит, уменьши `--batch_size` или используй GPU.

### 4. Проверь результаты
- После обучения файл `colorizer_weights.pth` появится в папке `TintoraAi`.
- Если обучение прервалось, проверь наличие файлов вроде `colorizer_weights_epochX.pth`.

## Оптимизация обучения
- **Для слабых компьютеров**:
  - Уменьши `--batch_size` до 2–4.
  - Уменьши `--num_classes` до 10–100.
  - Используй меньший датасет (100–500 пар изображений).
- **Для GPU**:
  - Проверь поддержку CUDA:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
  - Добавь смешанную точность для ускорения:
    ```python
    from torch.cuda.amp import autocast
    with autocast():
        # Вставь в цикл обучения в src/training/train.py
    ```
- **Мониторинг ресурсов**:
  - На GPU: `nvidia-smi`.
  - На CPU: Проверь память:
    ```bash
    python -c "import psutil; print(psutil.virtual_memory().used / 1024**3, 'GB')"
    ```

## Типичные проблемы и решения
- **"FileNotFoundError: Папка не найдена"**: Проверь, что папки `bw`, `color`, `labels` существуют и содержат файлы.
- **"ValueError: Image too small"**: Убедись, что все изображения ≥256x256.
- **"Out of memory"**: Уменьши `--batch_size` или используй GPU.
- **Ошибки с метками**: Проверь, что `.npy` файлы содержат числа от 0 до `num_classes-1`.

## Что делать дальше
- Протестируй модель с помощью [USAGE.md](USAGE.md):
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output colored.jpg --style modern
  ```
- Хочешь улучшить модель? Добавь больше данных или настрой параметры в `src/training/train.py`.
- Задай вопросы в [Issues](https://github.com/T1desLuck/TintoraAi/issues).

Удачи в обучении! Твои фото скоро станут цветными! 🎉