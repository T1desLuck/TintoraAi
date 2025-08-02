# Обучение TintoraAI

## Структура датасета

```
tintora_dataset/
├── color/   # Оригинальные цветные изображения
├── bw/      # Чёрно-белые версии
└── labels/  # Семантические метки (.npy)
```
Имена файлов совпадают (без расширения).

---

## Подготовка данных

1. **Создать папки:**
   ```bash
   mkdir -p tintora_dataset/color tintora_dataset/bw tintora_dataset/labels
   ```
2. **Собрать цветные изображения в `color/`.**
3. **Генерация ч/б копий:**
   ```python
   from PIL import Image
   import os
   color_dir = "tintora_dataset/color"
   bw_dir = "tintora_dataset/bw"
   os.makedirs(bw_dir, exist_ok=True)
   for filename in os.listdir(color_dir):
       if filename.endswith(('.jpg', '.png', '.jpeg')):
           img = Image.open(os.path.join(color_dir, filename))
           img.convert("L").save(os.path.join(bw_dir, filename))
   ```
4. **Генерация меток ([NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet)):**
   ```bash
   git clone https://github.com/T1desLuck/NpyLabelNet.git
   cd NpyLabelNet
   pip install -r requirements.txt
   python generate_labels.py --input ../TintoraAi/tintora_dataset/color --output ../TintoraAi/tintora_dataset/labels
   ```

---

## Конфигурация

Параметры обучения — в `config.yaml`:

```yaml
# Путь к данным
data_path: ./tintora_dataset

# Путь для сохранения моделей
save_path: models/colorizer_weights.pth

# Параметры обучения
epochs: 50               # Общее число эпох
batch_size: 8            # Размер батча
accum_steps: 4           # Шаги накопления градиентов 
lr: 0.0002               # Скорость обучения
num_classes: 100         # Число классов
patience: 10             # Ранняя остановка (эпох без улучшения)
save_frequency: 5        # Частота сохранения чекпоинтов (каждые N эпох)

# Параметры изображений
min_image_size: 64       # Минимальный размер изображений
max_image_size: 1024     # Максимальный размер изображений
pad_divisor: 16          # Делитель для padding
```

---

## Запуск обучения

### Локально

```bash
python src/training/train.py --config config.yaml
```

### Google Colab

```python
!python src/training/train.py --config config.yaml
```

### Vast.ai

```python
import os, subprocess
os.environ["TORCH_CUDA_ALLOC_RETRY"] = "1"
os.environ["TORCH_CUDA_MAX_SPLIT_SIZE_MB"] = "64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run(["/workspace/venv/bin/python", "/workspace/TintoraAI/src/training/train.py", "--config", "/workspace/TintoraAI/config.yaml"])
```

---

## Контрольные точки и чекпоинты

- **Лучшая модель**: `models/best_model.pth` (обновляется при улучшении валидационных метрик)
- **Финальная модель**: `models/colorizer_weights.pth` (сохраняется в конце обучения)
- **Регулярные чекпоинты**: `models/epoch{N}.pth` (сохраняются каждые `save_frequency` эпох)
- **Примеры колоризации**: `models/samples/` (если включено `save_samples: true`)

---

## Продолжение прерванного обучения

Если обучение было прервано (из-за сбоя, таймаута или другой причины), вы можете продолжить с последнего сохраненного чекпоинта.

### 1. Найдите последний чекпоинт

```python
import os
import re

def find_latest_checkpoint(models_dir="models"):
    checkpoints = []
    for filename in os.listdir(models_dir):
        if match := re.match(r'epoch(\d+)\.pth', filename):
            epoch = int(match.group(1))
            checkpoints.append((epoch, os.path.join(models_dir, filename)))
    
    return max(checkpoints, default=(0, None)) if checkpoints else (0, None)

epoch, checkpoint_path = find_latest_checkpoint()
print(f"Найден чекпоинт эпохи {epoch}: {checkpoint_path}")
```

### 2. Модифицируйте код train.py для загрузки чекпоинта

#### Локально 

Создайте скрипт `resume_training.py`:

```python
import torch
import os
import sys
from src.training.train import train
import argparse

parser = argparse.ArgumentParser(description="Продолжение обучения TintoraAI")
parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации")
parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту")
parser.add_argument("--start-epoch", type=int, required=True, help="Начальная эпоха")
args = parser.parse_args()

# Модифицируем train для загрузки весов и установки начальной эпохи
def resume_train():
    # Установка переменных окружения для продолжения обучения
    os.environ["RESUME_TRAINING"] = "1"
    os.environ["CHECKPOINT_PATH"] = args.checkpoint
    os.environ["START_EPOCH"] = str(args.start_epoch)
    
    # Запуск функции обучения
    train(args.config)

if __name__ == "__main__":
    resume_train()
```

Затем запустите:
```bash
python resume_training.py --checkpoint models/epoch25.pth --start-epoch 25 --config config.yaml
```

#### Google Colab

```python
# Добавьте в начало обучающего скрипта:
import os

# Установите переменные окружения для возобновления обучения
checkpoint_path = "/content/TintoraAi/models/epoch25.pth" 
start_epoch = 25

os.environ["RESUME_TRAINING"] = "1"
os.environ["CHECKPOINT_PATH"] = checkpoint_path
os.environ["START_EPOCH"] = str(start_epoch)

# Далее запустите train.py
!python src/training/train.py --config config.yaml
```

#### Vast.ai

```python
import os
import subprocess

# Настройка окружения
os.environ["TORCH_CUDA_ALLOC_RETRY"] = "1"
os.environ["TORCH_CUDA_MAX_SPLIT_SIZE_MB"] = "64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Установка переменных для возобновления обучения
os.environ["RESUME_TRAINING"] = "1"
os.environ["CHECKPOINT_PATH"] = "/workspace/TintoraAI/models/epoch25.pth"
os.environ["START_EPOCH"] = "25"

# Запуск обучения
subprocess.run([
    "/workspace/venv/bin/python", 
    "/workspace/TintoraAI/src/training/train.py", 
    "--config", "/workspace/TintoraAI/config.yaml"
])
```

### 3. Модификация train.py для поддержки возобновления обучения

Добавьте следующий код в функцию `train()` перед циклом обучения:

```python
# Проверка возобновления обучения
resume_training = os.environ.get("RESUME_TRAINING", "0") == "1"
if resume_training:
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    start_epoch = int(os.environ.get("START_EPOCH", "0"))
    
    if os.path.exists(checkpoint_path):
        # Загружаем состояние модели и оптимизаторов
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            # Полный чекпоинт со всеми состояниями
            model.load_state_dict(checkpoint["model_state"])
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D_state"]) 
            optimizer_C.load_state_dict(checkpoint["optimizer_C_state"])
            
            if "scheduler_G_state" in checkpoint:
                scheduler_G.load_state_dict(checkpoint["scheduler_G_state"])
                scheduler_D.load_state_dict(checkpoint["scheduler_D_state"])
                scheduler_C.load_state_dict(checkpoint["scheduler_C_state"])
                
            print(f"✅ Загружено полное состояние обучения из {checkpoint_path}")
        else:
            # Только веса модели
            model.load_state_dict(checkpoint)
            print(f"✅ Загружены только веса модели из {checkpoint_path}")
            
        print(f"Возобновление обучения с эпохи {start_epoch}")
        # Установка начальной эпохи
        epochs = epochs - start_epoch
    else:
        print(f"⚠️ Чекпоинт не найден: {checkpoint_path}")
```

### 4. Улучшенное сохранение чекпоинтов (сохраняйте полное состояние)

Замените код сохранения чекпоинтов на:

```python
# Сохраняем чекпоинт со всеми состояниями
if (epoch + 1) % config.get('save_frequency', 5) == 0:
    checkpoint_path = os.path.join(os.path.dirname(save_path), f'epoch{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_state': optimizer_D.state_dict(),
        'optimizer_C_state': optimizer_C.state_dict(),
        'scheduler_G_state': scheduler_G.state_dict(),
        'scheduler_D_state': scheduler_D.state_dict(),
        'scheduler_C_state': scheduler_C.state_dict(),
        'best_val_loss': best_val_loss,
        'no_improve_count': no_improve_count
    }, checkpoint_path)
    print(f"Чекпоинт сохранён: {checkpoint_path}")
```

---

## Важные файлы для обучения

1. **Весовые файлы** (необходимы для продолжения):
   - `models/epoch{N}.pth` — регулярные чекпоинты
   - `models/best_model.pth` — лучшая модель

2. **Файлы датасета** (необходимы для обучения):
   - Все файлы в директориях `tintora_dataset/bw`, `tintora_dataset/color` и `tintora_dataset/labels`

3. **Конфигурация** (можно модифицировать):
   - `config.yaml` — настройки обучения

4. **Код** (важны, но можно восстановить из репозитория):
   - `src/training/train.py`
   - `src/model/tintora_ai.py`
   - `src/training/dataset.py`
   - `src/training/gan_loss.py`

---

## Советы по обучению

- **Большие датасеты**: разделите данные на несколько частей и обучайтесь последовательно
- **Прерванное обучение**: всегда сохраняйте чекпоинты каждые несколько эпох
- **Vast.ai**: создавайте скрипт для автоматического скачивания весов на локальный компьютер
- **Google Colab**: копируйте чекпоинты на Google Drive, чтобы не потерять при отключении
- **Локально**: отслеживайте использование GPU памяти с помощью `nvidia-smi`

---

## Решение проблем

- **"expected scalar type Float but found Half"**: преобразуйте тензоры в `.float()` перед ssim.
- **Ошибка CrossEntropyLoss**: добавьте `.squeeze()` для labels.
- **Нет памяти**: уменьшайте batch_size, увеличивайте accum_steps.
- **Обучение останавливается**: проверьте логи на ошибки и продолжите с последнего чекпоинта.

---

## Ссылки

- TintoraAi: [github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)
- NpyLabelNet: [github.com/T1desLuck/NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet)
- Дата последнего обновления: 2025-08-02