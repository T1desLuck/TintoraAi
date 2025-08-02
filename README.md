# TintoraAI

TintoraAI — нейросеть для колоризации чёрно-белых изображений с использованием GAN и семантического анализа.

---

## Возможности

- Колоризация ч/б и выцветших фото
- Семантическая классификация (100 классов)
- Управление стилем (modern, vintage, sepia, dramatic)
- Пакетная обработка и настройка параметров цвета

---

## Быстрый старт

### Локально

```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Google Colab

```python
!git clone https://github.com/T1desLuck/TintoraAi.git
%cd TintoraAi
!pip install -r requirements.txt
```
Для загрузки данных используйте Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Vast.ai

1. Арендуйте GPU-инстанс с образом CUDA+Python.
2. Установите зависимости:
```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3. Для ускорения обучения добавьте в начале:
```python
import os
os.environ["TORCH_CUDA_ALLOC_RETRY"] = "1"
os.environ["TORCH_CUDA_MAX_SPLIT_SIZE_MB"] = "64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

## Архитектура

- Генератор — U-Net + Attention
- Дискриминатор — PatchGAN
- Классификатор — для семантики

---

## Документация

- [Установка](INSTALL.md)
- [Обучение](TRAINING.md)
- [Использование](USAGE.md)
- [Вклад](CONTRIBUTING.md)

---

## Ссылки

- Репозиторий: [github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)
- Генерация меток: [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet)