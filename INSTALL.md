# Установка TintoraAI

Этот документ описывает, как установить TintoraAI для локального запуска на ПК или в Google Colab без сторонних сервисов.

## Требования
- **Операционная система**: Windows, macOS, Linux (для ПК).
- **Python**: 3.8 или новее.
- **Аппаратное обеспечение**:
  - CPU: Любой современный процессор (4+ ядра для скорости).
  - GPU: Опционально (NVIDIA с CUDA).
  - ОЗУ: 8 ГБ (16 ГБ для обучения).
- **Интернет**: Только для установки зависимостей.

## Установка на ПК
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-username/TintoraAI.git
   cd TintoraAI
   ```

2. Создайте виртуальное окружение (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   Зависимости: `torch`, `torchvision`, `pillow`, `pytest`, `flake8`.

4. Проверка установки:
   ```bash
   python src/colorize.py --input examples/sample_images/bw/sample1.jpg
   ```
   Результат сохранится как `colored_image.jpg`.

## Установка в Google Colab
1. Откройте [Google Colab](https://colab.research.google.com).
2. Загрузите репозиторий:
   ```bash
   !git clone https://github.com/your-username/TintoraAI.git
   %cd TintoraAI
   ```
3. Установите зависимости:
   ```bash
   !pip install -r requirements.txt
   ```
4. Проверьте запуск:
   ```bash
   !python src/colorize.py --input examples/sample_images/bw/sample1.jpg
   ```
   Скачайте результат (`colored_image.jpg`) из Colab.

## Устранение неполадок
- **Ошибка CUDA**: Установите PyTorch с CUDA (`pip install torch torchvision`). Без GPU используется CPU.
- **Зависимости**: Обновите pip: `pip install --upgrade pip`.
- **Файл не найден**: Проверьте путь к входному фото.