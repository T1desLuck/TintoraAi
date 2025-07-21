# Contributing to TintoraAI

Спасибо, что хотите внести вклад в TintoraAI, созданный [T1desLuck](https://github.com/T1desLuck)! Следуйте этим шагам, чтобы начать.

## Установка для разработки

### Локально
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/T1desLuck/TintoraAi.git
   cd TintoraAi
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8
   ```

### В Google Colab
1. Откройте Colab: [Google Colab](https://colab.research.google.com/).
2. Выполните команды:
   ```python
   !git clone https://github.com/T1desLuck/TintoraAi.git
   %cd TintoraAi
   !pip install -r requirements.txt
   !pip install pytest flake8
   ```

## Создание изменений

1. Создайте ветку:
   ```bash
   git checkout -b feature/your-feature
   ```
   или в Colab:
   ```python
   !git checkout -b feature/your-feature
   ```

2. Внесите изменения в код или документацию.

3. Запустите линтер:
   - **Локально**:
     ```bash
     flake8 src/ tests/
     ```
   - **В Colab**:
     ```python
     !flake8 src/ tests/
     ```

4. Запустите тесты:
   - **Локально**:
     ```bash
     PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/
     ```
   - **В Colab**:
     ```python
     import os
     os.environ['PYTHONPATH'] = '/content/TintoraAi/src:' + os.environ.get('PYTHONPATH', '')
     !pytest tests/
     ```

5. Закоммитьте и отправьте изменения:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature
   ```

## Создание Pull Request

1. Откройте Pull Request на [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi).
2. Убедитесь, что CI (linting и тесты) проходит.
3. Ожидайте отзыв от [T1desLuck](https://github.com/T1desLuck).

## Структура проекта

- `src/`: Основной код (модель, предобработка, постобработка).
- `tests/`: Юнит-тесты.
- `docs/`: Документация (если есть).

## Тестирование

Добавляйте тесты в `tests/`. Убедитесь, что импорты используют `src.model.tintora_ai`.

## Код-стайл

Следуйте PEP 8. Используйте `flake8` для проверки.