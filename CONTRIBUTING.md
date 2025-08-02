# Вклад в TintoraAI

Спасибо за интерес к TintoraAI! Здесь ты найдёшь пошаговые инструкции для работы с проектом: как локально, так и через Google Colab и Vast.ai. Подходит для новичков и профессионалов.

---

## Кодекс поведения

- Уважай других участников и их вклад.
- Соблюдай нормы этикета GitHub, будь дружелюбен.

---

## Как начать

### 1. Форк и клонирование

```bash
# Сначала форкни репозиторий на свой GitHub (кнопка Fork).
git clone https://github.com/YOUR-USERNAME/TintoraAi.git
cd TintoraAi
git remote add upstream https://github.com/T1desLuck/TintoraAi.git
```

### 2. Установка зависимостей

#### Локально

```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

#### Google Colab

```python
!git clone https://github.com/YOUR-USERNAME/TintoraAi.git
%cd TintoraAi
!pip install -r requirements.txt
```

#### Vast.ai

```bash
git clone https://github.com/YOUR-USERNAME/TintoraAi.git
cd TintoraAi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Как внести изменения

1. **Создай ветку:**
   ```bash
   git checkout -b feature/имя-вашей-фичи
   # или
   git checkout -b fix/описание-исправления
   ```

2. **Внеси изменения в код или документацию.**
   - Следуй PEP8 (отступы — 4 пробела).
   - Документируй публичные функции и классы.
   - Старайся делать атомарные коммиты с осмысленными комментариями.

3. **Тестируй изменения:**
   ```bash
   pytest
   flake8 src/ tests/
   ```

---

## Стиль коммитов

- `feat:` — новая функциональность
- `fix:` — исправление ошибок
- `docs:` — документация
- `test:` — тесты
- `refactor:` — рефакторинг

Пример:
```
feat: добавил стиль sepia
fix: исправил ошибку загрузки модели
docs: обновил README с примерами Colab
```

---

## Pull Request

1. **Синхронизируй ветку:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Отправь изменения:**
   ```bash
   git push origin feature/имя-вашей-фичи
   ```
3. **Оформи pull request через GitHub:**
   - Описывай, что и зачем изменил.
   - Добавь ссылку на соответствующий Issue, если есть.

---

## Тестирование и документация

- Добавляй тесты для новых функций (в папку `tests/`, имя файла начинается с `test_`).
- Обновляй документацию: README.md, USAGE.md, TRAINING.md, INSTALL.md.
- Пример запуска тестов:
  ```bash
  pytest
  ```

---

## Структура проекта

```
TintoraAi/
├── src/
│   ├── model/
│   ├── training/
│   ├── colorize.py
├── tintora_dataset/
│   ├── bw/
│   ├── color/
│   └── labels/
├── models/
├── tests/
├── README.md
├── INSTALL.md
├── USAGE.md
├── TRAINING.md
├── CONTRIBUTING.md
```

---

## Сообщить об ошибке или предложить идею

- Создай Issue на GitHub с подробным описанием проблемы или предложения.
- Для багов указывай версию Python, ОС, шаги для воспроизведения, лог ошибки.
- Для новых идей — опиши, зачем это нужно и как это улучшит проект.

---

## Советы и лайфхаки

- **Локально:** всегда создавай виртуальное окружение.
- **Colab:** данные удобно хранить на Google Drive.
- **Vast.ai:** используй переменные окружения для ускорения работы с CUDA.
- **Новички:** не бойся задавать вопросы через Issue!
- **Профи:** не забывай про тесты и документацию для своих изменений.

---

## Обратная связь

- Основной репозиторий: [github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)
- Автор: T1desLuck
- Проект для генерации меток: [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet)

Спасибо за вклад в TintoraAI!