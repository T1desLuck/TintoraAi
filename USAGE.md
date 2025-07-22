# Использование TintoraAI

Готов раскрасить свои старые фото с помощью TintoraAI? Это просто! Следуй этим шагам, и у тебя всё получится, даже если ты новичок.

## Что нужно
- Установленный TintoraAI (смотри [INSTALL.md](INSTALL.md)).
- Обученная модель (смотри [TRAINING.md](TRAINING.md)) или файл `colorizer_weights.pth`.
- Чёрно-белое фото в формате `.jpg`.

## Шаги использования

### 1. Подготовь фото
- Найди чёрно-белое фото (например, `old_photo.jpg`).
- Положи его в любую папку, например, `photos/`.

### 2. Открой терминал
- Перейди в папку проекта:
  ```bash
  cd TintoraAi
  ```

### 3. Запусти раскраску
- Введи команду с путём к фото:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg
  ```
- По умолчанию результат сохранится как `colored_image.jpg` в той же папке.

### 4. Настрой параметры (по желанию)
- Измени имя выходного файла:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --output new_color.jpg
  ```
- Выбери стиль:
  - `--style modern` — яркие цвета.
  - `--style vintage` — приглушённые тона.
  - `--style neutral` — средний вариант.
  Пример:
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --style vintage
  ```
- Настрой насыщенность (от 0.5 до 2.0):
  ```bash
  python src/colorize.py --input photos/old_photo.jpg --saturation 1.5
  ```

### 5. Проверь результат
- Открой файл `colored_image.jpg` (или то имя, что ты указал).
- Если цвета странные, проверь, обучена ли модель, или попробуй другой стиль.

## Если что-то не работает
- Убедись, что файл `colorizer_weights.pth` есть в папке проекта.
- Если его нет, обучи модель (смотри [TRAINING.md](TRAINING.md)).
- Посмотри ошибки в терминале и спроси в [Issues](https://github.com/T1desLuck/TintoraAi/issues), если не разберёшься.

## Пример
Допустим, у тебя фото `photos/family.jpg`. Команда:
```bash
python src/colorize.py --input photos/family.jpg --output family_color.jpg --style modern --saturation 1.2
```
Результат: `family_color.jpg` с яркими цветами.

Попробуй и наслаждайся! 📸