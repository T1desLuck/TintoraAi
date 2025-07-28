# Обучение TintoraAI

Хочешь обучить **TintoraAI** на своих данных, чтобы раскрашивать черно-белые или выцветшие фото? Этот гайд поможет подготовить датасет и запустить обучение шаг за шагом. Инструкции подойдут как для новичков, так и для опытных разработчиков.

## Что нужно для обучения
- Установленный **TintoraAI** (см. [INSTALL.md](INSTALL.md)).
- Установленный **NpyLabelNet** для генерации `.npy` меток (см. ниже).
- Компьютер с процессором (CPU) или видеокартой (GPU NVIDIA с CUDA для ускорения).
- Датасет с тремя типами данных:
  - Черно-белые изображения (`bw/`, 512x512).
  - Цветные изображения (`color/`, 512x512).
  - Метки `.npy` (`labels/`, числа 0–99, созданные `NpyLabelNet`).
- Терпение: обучение может занять время, особенно на CPU.

## Установка NpyLabelNet
**NpyLabelNet** — это нейронная сеть для классификации изображений и создания `.npy` меток, необходимых для обучения **TintoraAI**. Она классифицирует изображения на 100 классов (например, "глаза", "дерево") и сохраняет ID классов в `.npy` файлы.

### Локальная установка
1. Клонируй репозиторий **NpyLabelNet**:
   ```bash
   git clone https://github.com/T1desLuck/NpyLabelNet.git
   cd NpyLabelNet
   ```
2. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Убедись, что у тебя есть файл `classes.json` (100 классов) и веса модели (`models/tiny_cnn.pth`).

## Подготовка датасета
**TintoraAI** ожидает датасет с тремя папками: `bw` (черно-белые изображения), `color` (цветные изображения) и `labels` (`.npy` файлы с метками). Все изображения должны быть 512x512 пикселей.

### 1. Собери изображения
- Найди пары изображений: черно-белое и его цветная версия. Размер **512x512 пикселей** обязателен.
- Источники:
  - Личные фотографии (отсканированные черно-белые фото и их цветные аналоги).
  - Открытые фотостоки: Unsplash, Pexels (убедись, что лицензия позволяет использование).
- Для тестов хватит 100–500 пар. Для хорошего результата — **10,000+ пар**.

### 2. Создай структуру папок
- Создай папку `tintora_dataset` в корне проекта **TintoraAI**:
  ```bash
  mkdir tintora_dataset
  mkdir tintora_dataset/bw
  mkdir tintora_dataset/color
  mkdir tintora_dataset/labels
  ```
- Положи файлы:
  - `tintora_dataset/bw/photo1.jpg` — черно-белое изображение (512x512).
  - `tintora_dataset/color/photo1.jpg` — цветная версия (512x512).
  - `tintora_dataset/labels/photo1.npy` — метка (например, `[0]` для "глаза").

### 3. Подготовь черно-белые изображения
- Если у тебя есть цветные изображения, но нет черно-белых, создай их:
  ```python
  from PIL import Image
  import os

  color_dir = "tintora_dataset/color"
  bw_dir = "tintora_dataset/bw"
  os.makedirs(bw_dir, exist_ok=True)

  for file in os.listdir(color_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          img = Image.open(os.path.join(color_dir, file)).convert("L")
          img.save(os.path.join(bw_dir, file))
  ```
- Сохрани как `convert_to_bw.py` и запусти:
  ```bash
  python convert_to_bw.py
  ```
- Проверь, что файлы в `bw/` и `color/` имеют одинаковые имена и размер 512x512.

### 4. Создай метки (`.npy`) с помощью NpyLabelNet
- Используй **NpyLabelNet** для создания `.npy` файлов с ID классов (0–99) на основе цветных изображений из папки `tintora_dataset/color`.
- Перейди в папку **NpyLabelNet**:
  ```bash
  cd /путь/к/NpyLabelNet
  ```
- Запусти инференс:
  ```bash
  python auto_label.py --input /путь/к/TintoraAi/tintora_dataset/color --output /путь/к/TintoraAi/tintora_dataset/labels --model_path models/tiny_cnn.pth --classes_path classes.json
  ```
- Результат: `.npy` файлы (например, `photo1.npy`) появятся в `tintora_dataset/labels/`. Каждый файл содержит одно число (например, `[0]` для "глаза").
- Логи будут выглядеть так: `Processed photo1.jpg: class 0 (глаза), confidence 0.85`.
- Если уверенность модели ниже 0.5, присваивается класс 99 ("неопределённый_объект").

### 5. Проверь датасет
- Убедись, что:
  - Файлы в `bw/`, `color/` и `labels/` имеют одинаковые имена (например, `photo1.jpg` и `photo1.npy`).
  - Изображения ровно 512x512 пикселей.
  - `.npy` файлы содержат числа от 0 до 99.
- Используй этот скрипт:
  ```python
  from PIL import Image
  import numpy as np
  import os

  bw_dir = "tintora_dataset/bw"
  color_dir = "tintora_dataset/color"
  label_dir = "tintora_dataset/labels"

  for file in os.listdir(bw_dir):
      if file.endswith(".jpg") or file.endswith(".png"):
          bw_img = Image.open(os.path.join(bw_dir, file))
          color_img = Image.open(os.path.join(color_dir, file))
          label = np.load(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")))
          assert bw_img.size == (512, 512), f"Size mismatch for {file}"
          assert color_img.size == (512, 512), f"Size mismatch for {file}"
          assert 0 <= label[0] <= 99, f"Invalid label {label[0]} for {file}"
  ```
- Сохрани как `check_dataset.py` и запусти:
  ```bash
  python check_dataset.py
  ```

## Запуск обучения TintoraAI

### 1. Перейди в папку проекта
```bash
cd /путь/к/TintoraAi
```

### 2. Настрой PYTHONPATH
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```
(На Windows используй `set` вместо `export`).

### 3. Запусти обучение
- Базовая команда:
  ```bash
  python src/train.py --data_path tintora_dataset --save_path models/colorizer_weights.pth
  ```
- Настраиваемые параметры:
  - `--epochs 20`: Количество эпох (20 для хорошего результата).
  - `--batch_size 8`: Размер батча (8 для GPU, 4 для CPU).
  - `--accum_steps 4`: Шаги накопления градиента для больших батчей.
- Пример команды:
  ```bash
  python src/train.py --data_path tintora_dataset --save_path models/colorizer_weights.pth --epochs 20 --batch_size 8 --accum_steps 4
  ```

### 4. Следи за процессом
- В терминале будут отображаться строки вида:
  ```
  Epoch 1/20, Color Loss: 0.8500, Class Loss: 2.3000, SSIM: 0.7500, LPIPS: 0.4000, Accuracy: 0.6000
  ```
- Меньшие потери (`Color Loss`, `Class Loss`, `LPIPS`) и большие `SSIM` и `Accuracy` указывают на лучшее обучение.
- Чекпоинты сохраняются каждые 5 эпох в `models/checkpoint_epoch_X.pth`.

### 5. Проверь результаты
- После обучения файл `models/colorizer_weights.pth` появится в папке `TintoraAi`.
- Используй веса для раскраски (см. [USAGE.md](USAGE.md)).

## Оптимизация обучения

### Для слабых компьютеров
- Уменьши `--batch_size` до 4 или 2.
- Уменьши `--epochs` до 10 для теста.
- Используй меньший датасет (100–500 пар).

### Для GPU
- Проверь поддержку CUDA:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Для ускорения добавь смешанную точность в `src/train.py`:
  ```python
  from torch.cuda.amp import autocast
  with autocast():
      color_output, semantic_output = model(bw_images)
  ```

### Мониторинг ресурсов
- На GPU: Используй `nvidia-smi`.
- На CPU: Проверь использование памяти:
  ```bash
  python -c "import psutil; print(psutil.virtual_memory().used / 1024**3, 'GB')"
  ```

## Типичные проблемы и решения
- **"FileNotFoundError: Папка не найдена"**:
  - Проверь, что папки `bw/`, `color/`, `labels/` существуют и содержат файлы.
- **"ValueError: Image size mismatch"**:
  - Убедись, что все изображения ровно 512x512.
  - Измени размер:
    ```bash
    convert image.jpg -resize 512x512! resized_image.jpg
    ```
- **"ValueError: Invalid label"**:
  - Проверь, что `.npy` файлы содержат числа 0–99.
  - Перегенерируй метки с помощью `NpyLabelNet`.
- **"Out of memory"**:
  - Уменьши `--batch_size` до 2–4.
  - Используй GPU.

## Что делать дальше
- Протестируй модель с помощью [USAGE.md](USAGE.md):
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output colored.jpg --style modern
  ```
- Для улучшения добавь больше данных или настрой параметры в `src/train.py`.
- Задавай вопросы в [Issues](https://github.com/T1desLuck/TintoraAi/issues).

Удачи в обучении! Твои черно-белые фото скоро засияют цветами! 🎉