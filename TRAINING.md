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

Параметры обучения — в `config.yaml`.

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

## Контрольные точки

- Лучшие веса: `models/best_model.pth`
- Финальные: `models/colorizer_weights.pth`
- Примеры: `models/samples/`

---

## Советы

- Большие изображения + слабый GPU: уменьшайте batch_size, увеличивайте accum_steps.
- Ошибка pytorch_msssim — добавьте `.float()` к ssim.
- Формат меток — `.npy` с int64.

---

## Решение проблем

- "expected scalar type Float but found Half": преобразуйте тензоры в `.float()` перед ssim.
- Ошибка CrossEntropyLoss: добавьте `.squeeze()` для labels.
- Нет памяти: уменьшайте batch_size, увеличивайте accum_steps.

---

## Ссылки

- TintoraAi: [github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)
- NpyLabelNet: [github.com/T1desLuck/NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet)