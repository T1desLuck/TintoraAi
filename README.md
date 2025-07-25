# TintoraAI — Раскраска чёрно-белых фотографий

Добро пожаловать в **TintoraAI**! Это нейросеть для автоматической раскраски чёрно-белых фотографий с использованием глубокого обучения. Она проста в использовании, полностью автономна и поддерживает разные стили. Даже если ты новичок, этот проект для тебя!

## Что такое TintoraAI?
TintoraAI — это программа, которая:
- Превращает чёрно-белые фотографии в цветные с помощью U-Net и GAN.
- Учитывает семантику изображения (распознаёт объекты, например, небо или траву).
- Поддерживает стили: `modern` (яркие цвета), `vintage` (приглушённые тона), `neutral` (стандартные цвета).
- Работает локально, без внешних сервисов или данных.
- Может обучаться на твоих собственных фотографиях.

## Возможности
- **Инференс**: Раскрашивай одно фото за раз с настройкой стиля и насыщенности.
- **Обучение**: Настрой модель на своём датасете (чёрно-белые и цветные фото + метки).
- **Стилизация**: Выбирай между современным, винтажным или нейтральным стилем.
- **Автономность**: Все веса и данные хранятся локально, никаких облачных API.

## Установка
Чтобы установить TintoraAI, следуй инструкциям в [INSTALL.md](INSTALL.md). Там описано, как установить Python 3.9, Git и зависимости.

## Использование
Хочешь раскрасить старое фото? Читай [USAGE.md](USAGE.md) — там всё объяснено, от подготовки фото до команды для раскраски.

## Обучение
Чтобы улучшить модель на своих данных, смотри [TRAINING.md](TRAINING.md). Ты узнаешь, как подготовить датасет и запустить обучение.

## Как подготовить датасет
Для обучения нужен датасет с тремя папками:
- `bw/`: Чёрно-белые изображения (`.jpg` или `.png`, минимум 256x256 пикселей).
- `color/`: Цветные версии тех же изображений (те же имена файлов).
- `labels/`: Метки объектов в формате `.npy` (числа от 0 до 999, например, из ImageNet).
Подробности и примеры — в [TRAINING.md](TRAINING.md). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15_geSdpK33UzI1ndn2nzQvIhXRggHAFv?usp=sharing)

## Как помочь проекту
Хочешь улучшить TintoraAI? Читай [CONTRIBUTING.md](CONTRIBUTING.md), чтобы узнать, как добавить код, тесты или документацию.

## Требования
- **Python 3.9** (как в CI/CD).
- Библиотеки из `requirements.txt` (устанавливаются автоматически).
- (Опционально) GPU с CUDA для быстрого обучения.

## Лицензия
Проект распространяется под [MIT License](LICENSE).

## Полезные ресурсы
- Загрузка изображений для датасета: Используй открытые источники, такие как Unsplash или Pexels. Пример скрипта для загрузки — в [TRAINING.md](TRAINING.md).
- Обсуждение и вопросы: [Issues](https://github.com/T1desLuck/TintoraAi/issues).

## Спасибо!
Спасибо, что интересуешься TintoraAI! Если есть вопросы или идеи, пиши в [Issues](https://github.com/T1desLuck/TintoraAi/issues). Давай сделаем раскраску старых фото ещё круче! 📸
