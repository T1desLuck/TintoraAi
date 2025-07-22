# Contributing to TintoraAI

Вносите вклад в TintoraAI от [T1desLuck](https://github.com/T1desLuck)!

## Установка для разработки
### Локально
```bash
git clone https://github.com/T1desLuck/TintoraAi.git
cd TintoraAi
pip install -r requirements.txt
pip install pytest flake8
```

### В Google Colab
```python
!git clone https://github.com/T1desLuck/TintoraAi.git
%cd TintoraAi
!pip install -r requirements.txt
!pip install pytest flake8
```

## Создание изменений
1. Ветка:
   - Локально: `git checkout -b feature/your-feature`
   - Colab: `!git checkout -b feature/your-feature`
2. Измените код.
3. Линтинг:
   - Локально: `flake8 src/ tests/`
   - Colab: `!flake8 src/ tests/`
4. Тесты:
   - Локально: `PYTHONPATH=$PYTHONPATH:$(pwd)/src pytest tests/`
   - Colab: `import os; os.environ['PYTHONPATH'] = '/content/TintoraAi/src:' + os.environ.get('PYTHONPATH', ''); !pytest tests/`
5. Коммит:
   ```bash
   git add .
   git commit -m "Add your feature"
   git push origin feature/your-feature
   ```

## Pull Request
1. Откройте PR на [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi).
2. Убедитесь, что CI проходит.
3. Свяжитесь с [T1desLuck](https://github.com/T1desLuck).

## Планы
- Добавление GAN для качества.
- Постобработка для реализма.

Репозиторий: [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)