# Использование TintoraAI

Готов раскрасить свои старые черно-белые фото с помощью **TintoraAI**? Это проще простого! Этот гайд покажет, как использовать программу, даже если ты никогда не работал с Python. Всё объяснено для новичков.

## Что нужно
- Установленный **TintoraAI** (см. [INSTALL.md](INSTALL.md)).
- Обученная модель (`models/colorizer_weights.pth`, см. [TRAINING.md](TRAINING.md)).
- Черно-белое фото в формате `.jpg` или `.png` (рекомендуемый размер 512x512 пикселей).
- Компьютер с Python 3.9 (GPU ускорит процесс, но CPU тоже подойдет).

## Шаги использования

### 1. Подготовь фото
- Найди черно-белое фото (например, `old_photo.jpg`).
- Положи его в любую папку, например, `photos/`.
- Убедись, что размер фото 512x512 пикселей для оптимального результата. Проверь:
  ```python
  from PIL import Image
  img = Image.open("photos/old_photo.jpg")
  print(img.size)  # Должно быть (512, 512)
  ```
- Если фото другого размера, измени его:
  ```bash
  convert old_photo.jpg -resize 512x512! resized_photo.jpg
  ```

### 2. Проверь наличие весов модели
- Убедись, что файл `models/colorizer_weights.pth` есть в папке `TintoraAi`.
- Если его нет, обучи модель (см. [TRAINING.md](TRAINING.md)).

### 3. Открой терминал
- Перейди в папку проекта:
  ```bash
  cd /путь/к/TintoraAi
  ```
- Настрой `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  ```
  (На Windows используй `set` вместо `export`).
- Если используешь виртуальное окружение, активируй его:
  - Windows:
    ```bash
    .\venv\Scripts\activate
    ```
  - Mac/Linux:
    ```bash
    source venv/bin/activate
    ```

### 4. Запусти раскраску
- Базовая команда:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg
  ```
- Результат сохранится как `colored_image.jpg` в папке `TintoraAi`.

### 5. Настрой параметры (по желанию)
- Измени имя выходного файла:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output my_photo.jpg
  ```
- Выбери стиль:
  - `--style modern`: Яркие, современные цвета.
  - `--style vintage`: Приглушенные тона, как на старых фото.
  - `--style neutral`: Стандартные цвета.
  Пример:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output vintage_photo.jpg --style vintage
  ```
- Настрой насыщенность цветов (0.5–2.0):
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --saturation 1.5
  ```
- Полный пример:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output family_color.jpg --style modern --saturation 1.2
  ```

### 6. Проверь результат
- Открой выходной файл (например, `family_color.jpg`) в любом просмотрщике изображений.
- Если цвета выглядят странно:
  - Проверь, обучена ли модель (`models/colorizer_weights.pth`).
  - Попробуй другой стиль или насыщенность.
  - Убедись, что входное фото 512x512.

### 7. Проверь производительность
- Если есть GPU, убедись, что оно используется:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- Если процесс медленный, убедись, что фото не слишком большое (>2000x2000).

## Типичные проблемы и решения
- **"FileNotFoundError: colorizer_weights.pth"**:
  - Обучи модель (см. [TRAINING.md](TRAINING.md)).
  - Проверь, что файл весов в папке `models/`.
- **"ValueError: Input image too small"**:
  - Убедись, что фото 512x512 пикселей.
  - Измени размер с помощью `convert` или редактора.
- **"ModuleNotFoundError"**:
  - Проверь, что ты в папке `TintoraAi`, виртуальное окружение активно и `PYTHONPATH` настроен.
  - Переустанови зависимости:
    ```bash
    pip install -r requirements.txt
    ```
- **Цвета выглядят неестественно**:
  - Попробуй `--style neutral` или уменьши `--saturation` (например, 0.8).
  - Проверь качество обучения модели (возможно, нужно больше данных).

## Примеры
1. Раскрасить фото в современном стиле:
   ```bash
   python src/colorize.py --input photos/family.jpg --output family_color.jpg --style modern --saturation 1.2
   ```
2. Раскрасить с нейтральными цветами:
   ```bash
   python src/colorize.py --input photos/old_house.jpg --output house_color.jpg --style neutral
   ```
3. Увеличить насыщенность для ярких цветов:
   ```bash
   python src/colorize.py --input photos/portrait.jpg --output portrait_color.jpg --style modern --saturation 1.5
   ```

## Что дальше
- Хочешь улучшить результаты? Обучи модель на своем датасете (см. [TRAINING.md](TRAINING.md)).
- Хочешь добавить новый стиль? Смотри [CONTRIBUTING.md](CONTRIBUTING.md).
- Есть вопросы? Пиши в [Issues](https://github.com/T1desLuck/TintoraAi/issues).

Наслаждайся цветными фото! 📸