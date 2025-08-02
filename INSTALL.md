# Установка TintoraAI

## Системные требования

- Python 3.9+
- PyTorch >= 1.13.0
- CUDA (опционально)
- 4+ ГБ RAM, 2+ ГБ дискового пространства

---

## Локальная установка

```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Установка в Google Colab

```python
!git clone https://github.com/T1desLuck/TintoraAi.git
%cd TintoraAi
!pip install -r requirements.txt
```
Для загрузки данных — Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Установка на Vast.ai

1. Арендуйте инстанс с CUDA GPU (например, RTX 3090/A6000).
2. Выполните:
```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
3. Для оптимальной работы добавьте:
```python
import os
os.environ["TORCH_CUDA_ALLOC_RETRY"] = "1"
os.environ["TORCH_CUDA_MAX_SPLIT_SIZE_MB"] = "64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

## Структура директорий

```bash
mkdir -p tintora_dataset/bw tintora_dataset/color tintora_dataset/labels models
```

---

## Проверка установки

```bash
python -c "import torch; print('CUDA доступен:', torch.cuda.is_available())"
```

---

## Типичные проблемы

- CUDA out of memory: уменьшите batch_size, увеличьте accum_steps.
- pytorch_msssim: используйте `.float()` перед ssim.
- В Colab: используйте ! перед командами терминала.
- В Vast.ai: убедитесь, что есть права записи.