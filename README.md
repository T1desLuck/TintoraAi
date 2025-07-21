# TintoraAI

TintoraAI — это автономный инструмент для раскраски чёрно-белых фотографий с использованием нейронных сетей, созданный пользователем [T1desLuck](https://github.com/T1desLuck). Проект поддерживает фотореализм, обработку изображений любого размера и обучение с нуля без серверов или сторонних сервисов.

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
2. Создайте новый ноутбук и выполните следующие команды:
   ```python
   !git clone https://github.com/T1desLuck/TintoraAi.git
   %cd TintoraAi
   !pip install -r requirements.txt
   ```

## Использование

Раскрасьте изображение с помощью скрипта `colorize.py`:
- **Локально**:
  ```bash
  python src/colorize.py --input examples/sample_images/bw/sample1.jpg
  ```
- **В Colab**:
  ```python
  !python src/colorize.py --input examples/sample_images/bw/sample1.jpg
  ```
- Параметры:
  - `--input`: Путь к чёрно-белому изображению.
  - `--output`: Путь для сохранения результата (по умолчанию `colored_image.jpg`).
  - `--saturation`: Уровень насыщенности (0.5–2.0, по умолчанию 1.0).
  - `--style`: Стиль раскраски (`neutral`, `modern`, `vintage`, по умолчанию `neutral`).

Результат сохранится в указанный файл.

## Обучение

Для обучения модели подготовьте датасет с парами чёрно-белых и цветных изображений.

### Локально
```bash
python src/training/train.py --data_path /path/to/dataset
```
- `--data_path`: Путь к директории с поддиректориями `bw/` и `color/`.
- `--epochs`: Количество эпох (по умолчанию 10).
- `--batch_size`: Размер батча (по умолчанию 8).

### В Google Colab
```python
!python src/training/train.py --data_path /content/TintoraAi/dataset
```
- Загрузите датасет в Colab (например, через Google Drive).

Веса модели сохранятся как `colorizer_weights.pth`.

## Тестирование

### Локально
```bash
PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/
```

### В Google Colab
```python
import os
os.environ['PYTHONPATH'] = '/content/TintoraAi/src:' + os.environ.get('PYTHONPATH', '')
!pytest tests/
```

## Контрибьютинг

Смотрите [CONTRIBUTING.md](CONTRIBUTING.md) для информации о контрибьютинге.

## Лицензия

[MIT License](LICENSE) (добавьте файл LICENSE, если отсутствует).