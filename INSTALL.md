# Установка TintoraAI

Установите TintoraAI, созданный [T1desLuck](https://github.com/T1desLuck), для раскраски чёрно-белых фотографий.

## Требования
- Python 3.8+
- Git
- Доступ к интернету

## Установка зависимостей

### Локально
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/T1desLuck/TintoraAi.git
   cd TintoraAi
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### В Google Colab
1. Откройте [Google Colab](https://colab.research.google.com/).
2. Выполните команды:
   ```python
   !git clone https://github.com/T1desLuck/TintoraAi.git
   %cd TintoraAi
   !pip install -r requirements.txt
   ```

## Проверка установки
Запустите тесты:
- Локально: `PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/`
- Colab: `import os; os.environ['PYTHONPATH'] = '/content/TintoraAi/src:' + os.environ.get('PYTHONPATH', ''); !pytest tests/`

Репозиторий: [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)