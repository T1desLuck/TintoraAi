# Обучение TintoraAI

Этот документ описывает, как собрать датасет и обучить TintoraAI с нуля для фотореалистичной раскраски на ПК или в Google Colab.

## Подготовка датасета
1. **Структура датасета**:
   - Создайте папку `dataset/`:
     ```
     dataset/
       ├── bw/        # Черно-белые изображения
       ├── color/     # Цветные версии тех же изображений
     ```
   - Имена файлов должны совпадать (например, `image1.jpg`).

2. **Сбор данных**:
   - Используйте открытые архивы (Flickr Commons, Unsplash) или свои фото.
   - Конвертируйте цветные фото в ч/б:
     ```python
     from PIL import Image
     import os

     color_dir = "dataset/color"
     bw_dir = "dataset/bw"
     os.makedirs(bw_dir, exist_ok=True)
     for img_name in os.listdir(color_dir):
         img = Image.open(os.path.join(color_dir, img_name)).convert("L")
         img.save(os.path.join(bw_dir, img_name))
     ```
   - Рекомендуемый размер: 1000+ пар изображений.

3. **Формат**:
   - Формат: JPG, JPEG, PNG.
   - Размеры: Любые (масштабируются до 256x256).

## Обучение на ПК
1. Запустите обучение:
   ```bash
   python src/training/train.py --data_path /path/to/dataset --epochs 10 --batch_size 8
   ```
   - `--data_path`: Путь к `dataset/`.
   - `--epochs`: Количество эпох (10–50).
   - `--batch_size`: 8 для GPU, 4 для CPU.

2. Требования:
   - GPU (NVIDIA с CUDA) или CPU.
   - ОЗУ: 16 ГБ+ для больших датасетов.

3. Результат:
   - Веса сохраняются в `colorizer_weights.pth`.
   - Добавьте их в `src/colorize.py`:
     ```python
     model.load_state_dict(torch.load("colorizer_weights.pth"))
     ```

## Обучение в Google Colab
1. Загрузите репозиторий:
   ```bash
   !git clone https://github.com/your-username/TintoraAI.git
   %cd TintoraAI
   ```
2. Установите зависимости:
   ```bash
   !pip install -r requirements.txt
   ```
3. Загрузите датасет (например, через Google Drive):
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Запустите обучение:
   ```bash
   !python src/training/train.py --data_path /content/drive/MyDrive/dataset --epochs 10
   ```

## Советы
- **Качество**: Используйте разнообразный датасет (портреты, пейзажи).
- **Скорость**: Для CPU уменьшите `batch_size` до 4.
- **Мониторинг**: Loss должен уменьшаться.

## Устранение неполадок
- **Out of Memory**: Уменьшите `batch_size` или размер изображений.
- **Плохое качество**: Увеличьте эпохи или данные.
- **Ошибки датасета**: Проверьте совпадение имен файлов.