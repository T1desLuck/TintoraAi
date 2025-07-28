# Как внести вклад в TintoraAI

Привет! Спасибо, что хочешь помочь с проектом **TintoraAI** — нейросетью для раскраски черно-белых фотографий! Этот файл объяснит, как ты можешь внести свой вклад, даже если ты новичок в программировании или GitHub. Мы рады всем, кто хочет улучшить проект!

## Что можно сделать
- Исправить ошибки в коде (например, баги в `src/colorize.py` или `src/train.py`).
- Добавить новые тесты в `tests/` (например, для полной модели `TintoraAI`).
- Улучшить документацию (`README.md`, `TRAINING.md`, этот файл и другие).
- Добавить новые стили в `src/colorize.py` (например, "cartoon" или "sepia").
- Оптимизировать производительность (например, добавить смешанную точность для GPU).
- Улучшить работу с `.npy` метками от `NpyLabelNet` (например, добавить валидацию).

## Шаги для участия

### 1. Прочитай документацию
- Ознакомься с файлами:
  - [README.md](README.md) — обзор проекта.
  - [INSTALL.md](INSTALL.md) — как установить.
  - [TRAINING.md](TRAINING.md) — как обучить модель.
  - [USAGE.md](USAGE.md) — как использовать.
- Убедись, что понимаешь, что ты хочешь сделать (исправить баг, добавить функцию, улучшить тесты).

### 2. Создай учетную запись на GitHub
- Если у тебя нет аккаунта, зарегистрируйся на [github.com](https://github.com) — это бесплатно!
- Войди в свой аккаунт и убедись, что можешь получить доступ к репозиторию [TintoraAi](https://github.com/T1desLuck/TintoraAi).

### 3. Найди задачу
- Зайди в раздел [Issues](https://github.com/T1desLuck/TintoraAi/issues).
- Найди задачу с меткой `help wanted` или `good first issue` — они подходят для новичков.
- Если хочешь предложить свою идею, создай новый Issue:
  - Нажми "New Issue".
  - Опиши, что хочешь сделать (например, "Добавить тест для полной модели TintoraAI").
  - Жди ответа от автора (@T1desLuck).

### 4. Согласуй задачу
- Напиши комментарий в выбранном Issue, чтобы сообщить, что ты берешься за задачу.
- Если создал новый Issue, убедись, что автор одобрил твою идею, чтобы избежать ненужной работы.

### 5. Настрой среду разработки
- Следуй инструкциям в [INSTALL.md](INSTALL.md), чтобы установить Python 3.9, Git и зависимости.
- Убедись, что можешь запустить тесты:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  pytest tests/ -s
  ```
- Если есть GPU, проверь поддержку CUDA:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  Должно вывести `True` при наличии GPU.

### 6. Создай ветку
- Склонируй репозиторий:
  ```bash
  git clone https://github.com/T1desLuck/TintoraAi.git
  cd TintoraAi
  ```
- Создай новую ветку с понятным названием (например, `fix-preprocess-bug` или `add-cartoon-style`):
  ```bash
  git checkout -b моя-новая-ветка
  ```

### 7. Сделай изменения
- Работай в папках `src/` (основной код), `tests/` (тесты) или документации (`*.md`).
- Если добавляешь функцию, создай тесты для нее в `tests/`. Например, для проверки `src/model/preprocess.py`:
  ```python
  from src.model.preprocess import preprocess_image
  from PIL import Image
  def test_preprocess():
      img = Image.new('L', (512, 512))
      tensor, size = preprocess_image(img)
      assert tensor.shape == (1, 1, 512, 512), "Wrong tensor shape"
      assert size == (512, 512), "Wrong size"
  ```
- Проверь стиль кода с помощью линтера:
  ```bash
  flake8 src/ tests/
  ```
  (Максимальная длина строки — 120 символов, см. `.flake8`).

### 8. Проверь свой код
- Запусти тесты локально:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  pytest tests/ -s
  ```
- Если добавляешь функционал в `src/colorize.py`, протестируй:
  ```bash
  python src/colorize.py --input tests/sample.jpg --output test_output.jpg --style neutral
  ```
- Убедись, что нет ошибок и код работает как ожидалось.

### 9. Зафиксируй изменения
- Добавь измененные файлы:
  ```bash
  git add .
  ```
- Создай коммит с понятным описанием, указав номер Issue (например, `#123`):
  ```bash
  git commit -m "Fix preprocess padding bug #123"
  ```

### 10. Отправь изменения
- Отправь ветку на GitHub:
  ```bash
  git push origin моя-новая-ветка
  ```

### 11. Создай Pull Request (PR)
- Зайди на [GitHub](https://github.com/T1desLuck/TintoraAi) и нажми "Compare & pull request".
- В описании PR укажи:
  - Что ты изменил (например, "Добавлен тест для полной модели TintoraAI").
  - Связь с Issue (например, "Closes #123").
- Жди, пока автор (@T1desLuck) проверит и одобрит PR. Возможно, потребуется внести правки.

## Правила и советы
- Используй **Python 3.9** (как в CI/CD).
- Следуй стилю PEP 8 (максимум 120 символов в строке, см. `.flake8`).
- Пиши тесты для новых функций в `tests/`.
- Если работаешь с `.npy` метками, убедись, что они содержат ID классов 0–99 от `NpyLabelNet`.
- Если работаешь с моделью, проверь производительность на GPU/CPU:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  ```
- Будь вежлив в комментариях и готов к обсуждению.

## Нужна помощь?
- Если что-то непонятно, создай Issue или напиши в существующем.
- Для вопросов по коду или идеям пиши автору (@T1desLuck).
- Если застрял с установкой, смотри [INSTALL.md](INSTALL.md) или спроси в Issue.

Спасибо за твой вклад! Ты делаешь TintoraAI лучше! 🚀