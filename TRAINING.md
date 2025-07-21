# Training TintoraAI

TintoraAI, созданный [T1desLuck](https://github.com/T1desLuck), можно обучить с нуля на собственном датасете. Следуйте этим шагам.

## Подготовка датасета

1. Создайте директорию с двумя поддиректориями:
   - `bw/`: Чёрно-белые изображения.
   - `color/`: Соответствующие цветные изображения (пары с `bw/`).

2. Убедитесь, что изображения имеют одинаковые размеры и формат (например, JPG).

Пример структуры:
```
dataset/
  bw/
    image1.jpg
    image2.jpg
  color/
    image1.jpg
    image2.jpg
```

## Установка

### Локально
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/T1desLuck/TintoraAi.git
   cd TintoraAi
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### В Google Colab
1. Откройте Colab: [Google Colab](https://colab.research.google.com/).
2. Выполните команды:
   ```python
   !git clone https://github.com/T1desLuck/TintoraAi.git
   %cd TintoraAi
   !pip install -r requirements.txt
   ```
3. Загрузите датасет (например, через Google Drive и смонтируйте его):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp -r /content/drive/MyDrive/dataset /content/TintoraAi/
   ```

## Обучение

### Локально
Запустите скрипт `train.py`:
```bash
python src/training/train.py --data_path /path/to/dataset
```
- `--data_path`: Путь к директории датасета.
- `--epochs`: Количество эпох (по умолчанию 10).
- `--batch_size`: Размер батча (по умолчанию 8).

### В Google Colab
```python
!python src/training/train.py --data_path /content/TintoraAi/dataset
```

### Выходные данные
- Веса модели сохраняются как `colorizer_weights.pth` в корневой директории.

## Мониторинг
Скрипт выводит текущую потерю (loss) для каждой эпохи. Для детального анализа добавьте логирование.

## Советы
- Используйте GPU (в Colab он доступен бесплатно).
- Увеличьте `--epochs` для лучшего качества при большом датасете.