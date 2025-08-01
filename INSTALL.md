# Установка TintoraAI

Этот документ объясняет, как установить и настроить проект `TintoraAI` для локального запуска, Google Colab или Vast.ai. Подходит для новичков и профессионалов.

## Требования
- **Операционная система**: Linux, Windows, macOS (или Google Colab/Vast.ai).
- **Python**: 3.9+.
- **GPU**: Рекомендуется с 16 ГБ видеопамяти (например, NVIDIA RTX 3060 или выше).
- **Диск**: Минимум 10 ГБ для датасета (25 000 изображений 512x512 и меток).
- **Зависимости**: Указаны в `requirements.txt`.

## Установка

### 1. Локальная установка
#### Для новичков
1. **Установите Python**:
   - Скачайте и установите Python 3.9+ с [python.org](https://www.python.org/downloads/).
   - Убедитесь, что `pip` доступен:
     ```bash
     python --version
     pip --version
     ```
2. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/<your-username>/TintoraAI.git
   cd TintoraAI
   ```
3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
   Это установит `torch`, `torchvision`, `pillow`, `pyyaml` и другие библиотеки.
4. **Подготовьте датасет**:
   - Создайте папку `tintora_dataset` в корне проекта.
   - Поместите в неё подпапки:
     - `bw/` — чёрно-белые изображения (.jpg).
     - `color/` — цветные изображения (.jpg).
     - `labels/` — метки (.npy), созданные [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git).
   - Убедитесь, что имена файлов совпадают, а метки в диапазоне 0–99.
5. **Настройте `config.yaml`**:
   - Откройте `config.yaml` в корне проекта.
   - Убедитесь, что `data_path` указывает на `./tintora_dataset`.
   - Пример:
     ```yaml
     data_path: ./tintora_dataset
     save_path: models/colorizer_weights.pth
     epochs: 20
     batch_size: 1
     accum_steps: 32
     lr: 0.001
     num_classes: 100
     pad_divisor: 16
     num_workers: 4
     ```

#### Для профессионалов
- Если у вас специфическая CUDA-версия, установите `torch` с поддержкой вашей версии:
  ```bash
  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
  ```
- Для ускорения загрузки данных уменьшите `num_workers` до 2 на слабых CPU.
- Проверьте доступность GPU:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

### 2. Google Colab
1. **Создайте новый ноутбук**:
   - Откройте [Google Colab](https://colab.research.google.com).
   - Выберите GPU в настройках: `Среда выполнения → Сменить тип среды → GPU`.
2. **Клонируйте репозиторий**:
   ```bash
   !git clone https://github.com/<your-username>/TintoraAI.git
   %cd TintoraAI
   ```
3. **Установите зависимости**:
   ```bash
   !pip install -r requirements.txt
   ```
4. **Загрузите датасет**:
   - Загрузите `tintora_dataset` на Google Диск.
   - Подключите диск в Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Обновите `config.yaml`:
     ```yaml
     data_path: /content/drive/MyDrive/tintora_dataset
     ```
5. **Запустите обучение**:
   ```bash
   !python src/training/train.py --config config.yaml
   ```

### 3. Vast.ai
1. **Создайте инстанс**:
   - Зарегистрируйтесь на [Vast.ai](https://vast.ai).
   - Выберите GPU с минимум 16 ГБ памяти (например, RTX 4000).
   - Установите шаблон с Python 3.9 и PyTorch.
2. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/<your-username>/TintoraAI.git
   cd TintoraAI
   ```
3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Загрузите датасет**:
   - Используйте SFTP или `scp` для загрузки `tintora_dataset` в инстанс.
   - Обновите `data_path` в `config.yaml` (например, `/root/tintora_dataset`).
5. **Запустите обучение**:
   ```bash
   python src/training/train.py --config config.yaml
   ```

## Проверка установки
- Проверьте зависимости:
  ```bash
  python -c "import torch, torchvision, PIL, yaml, pytorch_msssim; print('All good')"
  ```
- Проверьте датасет:
  ```bash
  ls tintora_dataset/bw | wc -l
  ls tintora_dataset/color | wc -l
  ls tintora_dataset/labels | wc -l
  ```
  Все три команды должны показать одинаковое количество файлов (например, 25000).

## Устранение ошибок
- **Нет GPU**: Убедитесь, что CUDA доступен (`torch.cuda.is_available()`).
- **Ошибки датасета**: Проверьте, что файлы в `bw/`, `color/`, `labels/` имеют одинаковые имена.
- **Нехватка памяти**: Уменьшите `num_workers` в `config.yaml` или выберите GPU с 24 ГБ.