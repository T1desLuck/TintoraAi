# Обучение TintoraAI

Обучайте TintoraAI от [T1desLuck](https://github.com/T1desLuck) для раскраски фото.

## Подготовка датасета
- Создайте директорию с поддиректориями:
  - `bw/`: Чёрно-белые изображения.
  - `color/`: Цветные пары.
- Включите дефекты (пятна, шум) для обучения.

## Установка
### Локально
```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
pip install -r requirements.txt
```

### В Google Colab
```python
!git clone https://github.com/T1desLuck/TintoraAi.git
%cd TintoraAi
!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/dataset /content/TintoraAi/
```

## Обучение
### Локально
```bash
python src/training/train.py --data_path /path/to/dataset
```
- `--epochs`: 10 (по умолчанию).
- `--batch_size`: 8 (по умолчанию).

### В Colab
```python
!python src/training/train.py --data_path /content/TintoraAi/dataset
```

## Планы
- Датасет 10k–100k пар.
- Добавление GAN для улучшения качества.
- Постобработка для кожи, глаз, губ.

Репозиторий: [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)