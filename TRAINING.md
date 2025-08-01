# Обучение TintoraAI

Этот документ описывает процесс обучения нейронной сети `TintoraAI` для раскраски изображений. Метки для датасета создаются с помощью [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git).

## Подготовка датасета
1. **Соберите изображения**:
   - Подготовьте 25 000 пар изображений (чёрно-белые и цветные) в формате `.jpg`.
   - Чёрно-белые изображения поместите в `tintora_dataset/bw/`.
   - Цветные — в `tintora_dataset/color/`.
   - Имена файлов должны совпадать (например, `image1.jpg` в обеих папках).
2. **Создайте метки**:
   - Используйте [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git) для генерации `.npy` меток.
   - Поместите метки в `tintora_dataset/labels/` (например, `image1.npy` для `image1.jpg`).
   - Метки должны быть в диапазоне 0–99 (100 классов).
3. **Проверьте структуру**:
   ```bash
   tree tintora_dataset/
   ```
   Ожидаемый вывод:
   ```
   tintora_dataset/
   ├── bw/
   │   ├── image1.jpg
   │   └── ...
   ├── color/
   │   ├── image1.jpg
   │   └── ...
   └── labels/
       ├── image1.npy
       └── ...
   ```

## Настройка конфигурации
1. Откройте `config.yaml` в корне проекта.
2. Убедитесь, что `data_path` указывает на папку `tintora_dataset`:
   ```yaml
   data_path: ./tintora_dataset
   save_path: models/colorizer_weights.pth
   epochs: 20
   batch_size: 1
   accum_steps: 32
   lr: 0.001
   num_classes: 100
   pad_divisor: 16
   num_workers: 4
   ```
3. При необходимости измените:
   - `save_path`: Путь для сохранения весов.
   - `epochs`: Количество эпох.
   - `lr`: Скорость обучения.
   - `num_workers`: Количество потоков (уменьшите до 2 на слабых CPU).

## Обучение

### Локально
1. Убедитесь, что зависимости установлены:
   ```bash
   pip install -r requirements.txt
   ```
2. Запустите обучение:
   ```bash
   python src/training/train.py --config config.yaml
   ```
3. Мониторинг:
   - Лоссы и SSIM выводятся в консоль после каждой эпохи.
   - Чекпоинты сохраняются каждые 5 эпох в `models/`.

### Google Colab
1. Создайте ноутбук и включите GPU:
   - `Среда выполнения → Сменить тип среды → GPU`.
2. Выполните команды:
   ```bash
   !git clone https://github.com/<your-username>/TintoraAI.git
   %cd TintoraAI
   !pip install -r requirements.txt
   ```
3. Загрузите датасет на Google Диск и подключите:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Обновите `config.yaml`:
   ```yaml
   data_path: /content/drive/MyDrive/tintora_dataset
   ```
5. Запустите:
   ```bash
   !python src/training/train.py --config config.yaml
   ```

### Vast.ai
1. Создайте инстанс с GPU (16 ГБ+).
2. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/<your-username>/TintoraAI.git
   cd TintoraAI
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Загрузите датасет:
   - Используйте `scp` или SFTP для передачи `tintora_dataset`.
   - Обновите `data_path` в `config.yaml`.
5. Запустите:
   ```bash
   python src/training/train.py --config config.yaml
   ```

## Устранение ошибок
- **CUDA out of memory**: Уменьшите `num_workers` или используйте GPU с 24 ГБ.
- **FileNotFoundError**: Проверьте пути в `config.yaml` и наличие файлов в `tintora_dataset`.
- **Ошибки меток**: Убедитесь, что `.npy` файлы созданы `NpyLabelNet` и содержат значения 0–99.

## Результаты
- После обучения веса сохраняются в `models/colorizer_weights.pth`.
- Используйте `src/colorize.py` для тестирования модели (см. `USAGE.md`).