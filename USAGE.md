# Использование TintoraAI

Этот документ объясняет, как использовать обученную модель `TintoraAI` для раскраски чёрно-белых изображений.

## Требования
- Установленный проект (см. `INSTALL.md`).
- Обученные веса модели (`models/colorizer_weights.pth`).
- Чёрно-белое изображение в формате `.jpg` или `.png`.

## Раскраска изображения
1. **Подготовьте изображение**:
   - Убедитесь, что изображение не меньше 256x256 пикселей.
2. **Запустите скрипт `colorize.py`**:
   ```bash
   python src/colorize.py --input path/to/bw_image.jpg --output colored_image.jpg
   ```
   - `--input`: Путь к чёрно-белому изображению.
   - `--output`: Путь для сохранения результата (по умолчанию `colored_image.jpg`).
   - `--saturation`: Насыщенность цветов (0.5–2.0, по умолчанию 1.0).
   - `--style`: Стиль раскраски (`neutral`, `modern`, `vintage`).
3. **Пример команды**:
   ```bash
   python src/colorize.py --input test.jpg --output result.jpg --saturation 1.2 --style vintage
   ```

### В Google Colab
1. Загрузите изображение:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
2. Запустите:
   ```bash
   !python src/colorize.py --input test.jpg --output result.jpg
   ```
3. Скачайте результат:
   ```python
   files.download('result.jpg')
   ```

### На Vast.ai
1. Загрузите изображение в инстанс:
   ```bash
   scp local_image.jpg user@vast_instance:/root/TintoraAI/test.jpg
   ```
2. Запустите:
   ```bash
   python src/colorize.py --input test.jpg --output result.jpg
   ```
3. Скачайте результат:
   ```bash
   scp user@vast_instance:/root/TintoraAI/result.jpg .
   ```

## Устранение ошибок
- **FileNotFoundError (веса)**: Убедитесь, что `colorizer_weights.pth` лежит в корне или укажите путь в `colorize.py`.
- **Ошибки размера**: Изображение должно быть не меньше 256x256.
- **Качество раскраски**: Если результат неудовлетворительный, проверьте обучение (см. `TRAINING.md`).

## Примечание
Метки для обучения создаются с помощью [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git), что обеспечивает семантическую точность раскраски.