# Установка TintoraAI

Хочешь запустить **TintoraAI** на своём компьютере? Этот гайд поможет установить всё шаг за шагом, даже если ты никогда не работал с Python или Git. Всё просто, следуй инструкциям!

## Что тебе понадобится
- Компьютер с **Windows**, **Mac** или **Linux**.
- **Python 3.9** (скачать можно с [python.org](https://www.python.org/downloads/release/python-390/)).
- **Git** (для скачивания кода, скачать с [git-scm.com](https://git-scm.com/)).
- (Опционально) Видеокарта (GPU) с поддержкой CUDA для ускорения обучения.
- Интернет для загрузки зависимостей.

## Шаги установки

### 1. Установи Python 3.9
- Скачай Python 3.9 с [официального сайта](https://www.python.org/downloads/release/python-390/).
- Во время установки на Windows:
  - Поставь галочку "Add Python to PATH".
  - Выбери "Customize installation" и убедись, что `pip` включён.
- Проверь установку в терминале (на Windows — "Командная строка" или PowerShell, на Mac/Linux — "Терминал"):
  ```bash
  python --version
  ```
  Должно вывести `Python 3.9.x`. Если нет, установи Python 3.9 и добавь его в PATH (поиск в Google: "Add Python to PATH").
- Проверь `pip`:
  ```bash
  pip --version
  ```
  Если выдаёт ошибку, обнови `pip`:
  ```bash
  python -m pip install --upgrade pip
  ```

### 2. Установи Git
- Скачай и установи Git с [git-scm.com](https://git-scm.com/).
- Проверь установку:
  ```bash
  git --version
  ```
  Должно вывести что-то вроде `git version 2.x.x`. Если не работает, переустанови Git.

### 3. Склонируй проект
- Открой терминал и введи:
  ```bash
  git clone https://github.com/T1desLuck/TintoraAi.git
  cd TintoraAi
  ```
- Это создаст папку `TintoraAi` с кодом проекта. Проверь содержимое:
  ```bash
  dir  # на Windows
  ls   # на Mac/Linux
  ```
  Должны быть видны файлы вроде `src/`, `tests/`, `requirements.txt`.

### 4. Создай виртуальное окружение (рекомендуется)
- Чтобы избежать конфликтов с другими проектами, создай виртуальное окружение:
  ```bash
  python -m venv venv
  ```
- Активируй его:
  - Windows:
    ```bash
    venv\Scripts\activate
    ```
  - Mac/Linux:
    ```bash
    source venv/bin/activate
    ```
- Убедись, что в терминале появилась надпись `(venv)`.

### 5. Установи зависимости
- В папке `TintoraAi` установи библиотеки:
  ```bash
  pip install -r requirements.txt
  ```
- Если возникают ошибки, проверь:
  - Активно ли виртуальное окружение (`(venv)` в терминале).
  - Работает ли интернет.
  - Версия `pip`:
    ```bash
    pip --version
    ```
    Если нужно, обнови:
    ```bash
    pip install --upgrade pip
    ```

### 6. Проверь поддержку GPU (если есть видеокарта)
- Если у тебя есть GPU NVIDIA, убедись, что `torch` поддерживает CUDA:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  - `True`: GPU работает, обучение будет быстрее.
  - `False`: Будет использоваться CPU (медленнее, но всё работает).
- Если GPU не работает, проверь драйверы NVIDIA и CUDA (поиск в Google: "Install CUDA for PyTorch").

### 7. Проверь установку
- Запусти тесты:
  ```bash
  pytest tests/
  ```
- Если видишь что-то вроде:
  ```
  ============================= 1 passed in 1.52s ===============================
  ```
  Установка успешна!
- Если есть ошибки, скопируй их текст и создай Issue на [GitHub](https://github.com/T1desLuck/TintoraAi/issues).

## Типичные проблемы и решения
- **"python: command not found"**: Python не добавлен в PATH. Переустанови Python с галочкой "Add to PATH".
- **"pip: command not found"**: Обнови `pip` или переустанови Python.
- **"ModuleNotFoundError"**: Убедись, что ты в папке `TintoraAi` и виртуальное окружение активно.
- **Ошибки с `torch` или CUDA**: Проверь, что у тебя правильная версия `torch` для GPU (см. [pytorch.org](https://pytorch.org/get-started/locally/)).

## Что дальше
- Узнай, как раскрасить фото, в [USAGE.md](USAGE.md).
- Хочешь обучить модель? Читай [TRAINING.md](TRAINING.md).
- Хочешь помочь проекту? Смотри [CONTRIBUTING.md](CONTRIBUTING.md).

Готово! Ты настроил TintoraAI! 🚀