# Использование TintoraAI

Этот документ описывает, как использовать TintoraAI для раскраски фотографий локально на ПК или в Google Colab.

## Использование на ПК
1. Запустите скрипт:
   ```bash
   python src/colorize.py --input path/to/image.jpg --output colored_image.jpg --saturation 1.2 --style modern
   ```
   - `--input`: Путь к ч/б или выцветшему фото (jpg, jpeg, png).
   - `--output`: Путь для сохранения результата (по умолчанию `colored_image.jpg`).
   - `--saturation`: Насыщенность цветов (0.5–2.0, по умолчанию 1.0).
   - `--style`: Стиль ("modern", "vintage", "neutral").

2. Пример:
   ```bash
   python src/colorize.py --input examples/sample_images/bw/sample1.jpg --saturation 1.5 --style vintage
   ```
   Результат сохранится как `colored_image.jpg`.

## Использование в Google Colab
1. Загрузите репозиторий и зависимости (см. [INSTALL.md](INSTALL.md)).
2. Загрузите фото в Colab (например, через панель файлов или Google Drive).
3. Запустите скрипт:
   ```bash
   !python src/colorize.py --input /content/sample.jpg --saturation 1.2 --style modern
   ```
4. Скачайте результат (`colored_image.jpg`) из Colab.

## Примечания
- Для фотореализма обучите модель (см. [TRAINING.md](TRAINING.md)).
- Поддерживаются любые размеры изображений без потери качества.
- Оригинал не изменяется.
- Если веса модели отсутствуют, результат будет случайным. Обучите модель для качества.