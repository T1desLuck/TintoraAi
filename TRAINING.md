# Обучение TintoraAI

Хочешь обучить **TintoraAI** на своих данных, чтобы раскрашивать черно-белые или выцветшие фото? Этот гайд поможет подготовить датасет и запустить обучение шаг за шагом. Инструкции подойдут как для новичков, так и для опытных разработчиков, и включают запуск как на локальном компьютере, так и в Google Colab. Проект полностью независим от сторонних данных и весов — используем только свои изображения и свои нейронки.

## Что нужно для обучения

- Установленный **TintoraAI** (см. [INSTALL.md](INSTALL.md)).
- Установленный **NpyLabelNet** для генерации `.npy` меток (см. ниже).
- Компьютер с процессором (CPU) или видеокартой (GPU NVIDIA с CUDA для ускорения, автоматически доступно в Colab).
- Датасет с тремя типами данных:
  - Черно-белые изображения (`bw/`).
  - Цветные изображения (`color/`).
  - Метки `.npy` (`labels/`), созданные с помощью **NpyLabelNet**.
- Терпение: обучение может занять время, особенно на CPU.

## Установка NpyLabelNet

**NpyLabelNet** — это нейронная сеть для классификации изображений и создания `.npy` меток, необходимых для обучения **TintoraAI**. Она классифицирует изображения на 1000 классов (например, "трава_лужайка", "человек_лицо") и сохраняет ID классов в `.npy` файлы.

### Локальная установка
1. Клонируйте репозиторий **NpyLabelNet**:
   ```bash
   git clone https://github.com/T1desLuck/NpyLabelNet.git
   cd NpyLabelNet
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Убедитесь, что у вас есть файл `classes.json` (1000 классов) и обученные веса модели (`models/tiny_cnn.pth`).

### Установка в Google Colab
1. Откройте Google Colab и создайте новый ноутбук.
2. Клонируйте репозиторий:
   ```python
   !git clone https://github.com/T1desLuck/NpyLabelNet.git
   %cd NpyLabelNet
   ```
3. Установите зависимости:
   ```python
   !pip install -r requirements.txt
   ```
4. Загрузите `classes.json` и `models/tiny_cnn.pth` в Colab (например, через Google Drive или `Files > Upload`).
5. (Опционально) Включите GPU: `Runtime > Change runtime type > GPU`.

## Подготовка датасета

**TintoraAI** ожидает датасет с тремя папками: `bw` (черно-белые изображения), `color` (цветные изображения) и `labels` (`.npy` файлы с метками). Все данные должны быть вашими — никаких сторонних наборов или весов.

### 1. Соберите изображения
- Найдите пары изображений: черно-белое и его цветная версия. Минимальный размер — **256x256 пикселей** (рекомендуется 512x512 для лучшей совместимости с **NpyLabelNet**).
- Источники:
  - Личные фотографии (отсканированные черно-белые фото и их цветные аналоги).
  - Открытые фотостоки: Unsplash, Pexels (убедитесь, что изображения ваши или лицензия позволяет использование).
- Для тестов хватит 100–500 пар изображений. Для хорошего результата — **10,000+ пар**.

### 2. Создайте структуру папок
- Создайте папку `dataset` в корне проекта **TintoraAI**:
  ```bash
  mkdir dataset
  mkdir dataset/bw
  mkdir dataset/color
  mkdir dataset/labels
  ```
- В Google Colab:
  ```python
  import os
  os.makedirs('dataset/bw', exist_ok=True)
  os.makedirs('dataset/color', exist_ok=True)
  os.makedirs('dataset/labels', exist_ok=True)
  ```
- Положите файлы:
  - `dataset/bw/photo1.jpg` — черно-белое изображение.
  - `dataset/color/photo1.jpg` — цветная версия того же изображения.
  - `dataset/labels/photo1.npy` — файл с меткой (например, `[150]` для "человек_лицо").

### 3. Подготовьте черно-белые изображения
- Если у вас есть цветные изображения, но нет черно-белых, создайте их:

#### Локально
```bash
convert dataset/color/photo1.jpg -colorspace Gray dataset/bw/photo1.jpg
```

#### В Google Colab
```python
from PIL import Image
import os

color_dir = "dataset/color"
bw_dir = "dataset/bw"
os.makedirs(bw_dir, exist_ok=True)

for file in os.listdir(color_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = Image.open(os.path.join(color_dir, file)).convert("L")
        img.save(os.path.join(bw_dir, file))
```
- Сохраните код как `convert_to_bw.py` и запустите:
  - Локально: `python convert_to_bw.py`
  - В Colab: `!python convert_to_bw.py`
- Проверьте, что файлы в `bw/` и `color/` имеют одинаковые имена и размеры (≥256x256).

### 4. Создайте метки (`.npy`) с помощью NpyLabelNet
- Используйте **NpyLabelNet** для создания `.npy` файлов с ID классов (0–999) на основе цветных изображений из папки `dataset/color`.

#### Локально
1. Перейдите в папку **NpyLabelNet**:
   ```bash
   cd /путь/к/NpyLabelNet
   ```
2. Запустите инференс:
   ```bash
   python auto_label.py --input /путь/к/TintoraAi/dataset/color --output /путь/к/TintoraAi/dataset/labels --model_path models/tiny_cnn.pth --classes_path classes.json
   ```
   - Замените `/путь/к/TintoraAi` на путь к проекту **TintoraAI**.
   - Убедитесь, что `models/tiny_cnn.pth` и `classes.json` доступны.

#### В Google Colab
1. Убедитесь, что **NpyLabelNet** клонирован и зависимости установлены (см. "Установка в Google Colab").
2. Загрузите цветные изображения, `classes.json` и `models/tiny_cnn.pth` в Google Drive (например, в `/content/drive/MyDrive/TintoraAi`).
3. Запустите инференс:
   ```python
   %cd /content/NpyLabelNet
   !python auto_label.py --input /content/drive/MyDrive/TintoraAi/dataset/color --output /content/drive/MyDrive/TintoraAi/dataset/labels --model_path /content/drive/MyDrive/models/tiny_cnn.pth --classes_path /content/drive/MyDrive/classes.json
   ```
- Результат: `.npy` файлы (например, `photo1.npy`) появятся в `dataset/labels/`. Каждый файл содержит одно число (ID класса, например, `[150]` для "человек_лицо").
- Логи будут выглядеть так: `Processed photo1.jpg: class 150 (человек_лицо), confidence 0.85`.
- Если уверенность модели ниже 0.5, присваивается класс 999 ("неопределённый_объект").

### 5. Проверьте датасет
- Убедитесь, что:
  - Все файлы в `bw/`, `color/` и `labels/` имеют одинаковые имена (например, `photo1.jpg` и `photo1.npy`).
  - Изображения имеют размер ≥256x256.
  - `.npy` файлы содержат числа от 0 до 999 (или меньше, если вы используете меньше классов).

#### Локально
```python
from PIL import Image
import numpy as np
import os

bw_dir = "dataset/bw"
color_dir = "dataset/color"
label_dir = "dataset/labels"

for file in os.listdir(bw_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        bw_img = Image.open(os.path.join(bw_dir, file))
        color_img = Image.open(os.path.join(color_dir, file))
        label = np.load(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")))
        assert bw_img.size == color_img.size, f"Size mismatch for {file}"
        assert bw_img.size[0] >= 256 and bw_img.size[1] >= 256, f"Image {file} too small"
        assert 0 <= label[0] < 1000, f"Invalid label for {file}"
```
- Сохраните как `check_dataset.py` и запустите:
  ```bash
  python check_dataset.py
  ```

#### В Google Colab
```python
from PIL import Image
import numpy as np
import os

bw_dir = "/content/drive/MyDrive/TintoraAi/dataset/bw"
color_dir = "/content/drive/MyDrive/TintoraAi/dataset/color"
label_dir = "/content/drive/MyDrive/TintoraAi/dataset/labels"

for file in os.listdir(bw_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        bw_img = Image.open(os.path.join(bw_dir, file))
        color_img = Image.open(os.path.join(color_dir, file))
        label = np.load(os.path.join(label_dir, file.replace(".jpg", ".npy").replace(".png", ".npy")))
        assert bw_img.size == color_img.size, f"Size mismatch for {file}"
        assert bw_img.size[0] >= 256 and bw_img.size[1] >= 256, f"Image {file} too small"
        assert 0 <= label[0] < 1000, f"Invalid label for {file}"
```
- Сохраните как `check_dataset.py` и запустите:
  ```python
  !python check_dataset.py
  ```

## Запуск обучения TintoraAI

### 1. Перейдите в папку проекта
#### Локально
```bash
cd /путь/к/TintoraAi
```

#### В Google Colab
```python
%cd /content/drive/MyDrive/TintoraAi
```

### 2. Запустите обучение
- Базовая команда:
  #### Локально
  ```bash
  python src/training/train.py --data_path dataset
  ```
  #### В Google Colab
  ```python
  !python src/training/train.py --data_path /content/drive/MyDrive/TintoraAi/dataset
  ```

- Настраиваемые параметры:
  - `--epochs 10`: Количество эпох (10 для теста, 50+ для хорошего результата).
  - `--batch_size 4`: Размер батча (4 для слабых GPU/CPU, 8–16 для мощных).
  - `--num_classes 100`: Количество классов для семантического анализа (100 для теста, 1000 по умолчанию).
- Пример команды:
  #### Локально
  ```bash
  python src/training/train.py --data_path dataset --epochs 10 --batch_size 4 --num_classes 100
  ```
  #### В Google Colab
  ```python
  !python src/training/train.py --data_path /content/drive/MyDrive/TintoraAi/dataset --epochs 10 --batch_size 4 --num_classes 100
  ```

### 3. Следите за процессом
- В терминале будут отображаться строки вида:
  ```
  Epoch 1/10, Loss: 0.85
  ```
- Меньший `Loss` указывает на лучшее обучение.
- Если обучение тормозит, уменьшите `--batch_size` или используйте GPU.

### 4. Проверьте результаты
- После обучения файл `colorizer_weights.pth` появится в папке `TintoraAi`.
- Если обучение прервалось, проверьте наличие промежуточных весов, например, `colorizer_weights_epochX.pth`.

## Оптимизация обучения

### Для слабых компьютеров
- Уменьшите `--batch_size` до 2–4.
- Уменьшите `--num_classes` до 10–100.
- Используйте меньший датасет (100–500 пар изображений).

### Для GPU
- Проверьте поддержку CUDA:
  #### Локально
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  #### В Google Colab
  ```python
  import torch
  print(torch.cuda.is_available())
  ```
- Для ускорения добавьте смешанную точность в `src/training/train.py`:
  ```python
  from torch.cuda.amp import autocast
  with autocast():
      # Вставьте в цикл обучения
  ```

### Мониторинг ресурсов
- На GPU: Используйте `nvidia-smi` (локально) или проверьте в Colab (`Runtime > View resources`).
- На CPU: Проверьте использование памяти:
  #### Локально
  ```bash
  python -c "import psutil; print(psutil.virtual_memory().used / 1024**3, 'GB')"
  ```
  #### В Google Colab
  ```python
  import psutil
  print(psutil.virtual_memory().used / 1024**3, 'GB')
  ```

## Типичные проблемы и решения

- **"FileNotFoundError: Папка не найдена"**:
  - Проверьте, что папки `bw/`, `color/`, `labels/` существуют и содержат файлы.
  - В Colab убедитесь, что Google Drive подключен (`drive.mount('/content/drive')`).
- **"ValueError: Image too small"**:
  - Убедитесь, что все изображения ≥256x256.
  - Измените размер:
    #### Локально
    ```bash
    convert image.jpg -resize 512x512! resized_image.jpg
    ```
    #### В Google Colab
    ```python
    from PIL import Image
    img = Image.open('image.jpg').resize((512, 512))
    img.save('resized_image.jpg')
    ```
- **"Out of memory"**:
  - Уменьшите `--batch_size` до 2–4.
  - Используйте GPU или Colab с включенным GPU.
- **Ошибки с метками**:
  - Проверьте, что `.npy` файлы содержат числа от 0 до `num_classes-1`.
  - Убедитесь, что имена файлов в `labels/` совпадают с `bw/` и `color/`.

## Что делать дальше

- Протестируйте модель с помощью [USAGE.md](USAGE.md):
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output colored.jpg --style modern
  ```
- Для улучшения модели добавьте больше данных или настройте параметры в `src/training/train.py`.
- Задавайте вопросы в [Issues](https://github.com/T1desLuck/TintoraAi/issues).

Удачи в обучении! Твои черно-белые фото скоро засияют цветами! 🎉