# Обучение TintoraAI

Этот документ описывает, как собрать датасет и обучить TintoraAI с нуля для достижения фотореалистичных результатов.

## Подготовка датасета
1. **Структура датасета**:
   - Создайте папку `dataset/` с двумя подпапками:
     ```
     dataset/
       ├── bw/        # Черно-белые изображения
       ├── color/     # Цветные версии тех же изображений
     ```
   - Имена файлов в `bw/` и `color/` должны совпадать (например, `image1.jpg`).

2. **Сбор данных**:
   - Используйте открытые архивы (Flickr Commons, Unsplash) или собственные цветные фото.
   - Конвертируйте цветные фото в ч/б для папки `bw/`:
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
   - Рекомендуемый размер датасета: 1000+ пар изображений для качественного обучения.

3. **Формат**:
   - Формат изображений: JPG, JPEG, PNG.
   - Размеры: Любые (модель автоматически масштабирует до 256x256).

## Обучение
1. **Запустите обучение**:
   ```bash
   python src/training/train.py --data_path /path/to/dataset --epochs 10 --batch_size 8
   ```
   - `--data_path`: Путь к папке `dataset/`.
   - `--epochs`: Количество эпох (рекомендуется 10–50).
   - `--batch_size`: Размер батча (8 для GPU с 4 ГБ, уменьшите для CPU).

2. **Требования**:
   - GPU (рекомендуется NVIDIA с CUDA) или CPU.
   - ОЗУ: 16 ГБ+ для больших датасетов.
   - PyTorch с поддержкой CUDA (если GPU).

3. **Процесс**:
   - Модель обучается с нуля, используя MSE и perceptual loss (VGG16) для фотореализма.
   - Веса сохраняются в `colorizer_weights.pth`.

4. **Интеграция весов**:
   После обучения добавьте веса в `src/ui/app.py` и `src/api/app.py`:
   ```python
   model.load_state_dict(torch.load("colorizer_weights.pth"))
   ```

## Обучение в Google Colab
1. Загрузите репозиторий:
   ```bash
   !git clone https://github.com-yours-username/TintoraAI.git
   %cd TintoraAI
   ```
2. Установите зависимости:
   ```bash
   !pip install -r requirements.txt
   ```
3. Загрузите датасет в Colab (например, через Google Drive).
4. Запустите обучение:
   ```bash
   !python src/training/train.py --data_path /content/dataset --epochs 10
   ```

## Советы
- **Качество**: Для фотореализма используйте датасет с разнообразными сценами (портреты, пейзажи, объекты).
- **Скорость**: Уменьшите `batch_size` до 4 для слабых GPU или CPU.
- **Мониторинг**: Следите за значением Loss в консоли (должно уменьшаться).

## Устранение неполадок
- **Out of Memory**: Уменьшите `batch_size` или размер изображений.
- **Плохое качество**: Увеличьте количество эпох или добавьте больше данных.
- **Ошибки датасета**: Убедитесь, что имена файлов в `bw/` и `color/` совпадают.