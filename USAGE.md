# Использование TintoraAI

Раскрашивайте чёрно-белые фотографии с помощью TintoraAI от [T1desLuck](https://github.com/T1desLuck).

## Запуск
Используйте скрипт `src/colorize.py` с параметрами.

### Локально
```bash
python src/colorize.py --input examples/sample_images/bw/sample1.jpg
```
- `--input`: Путь к ч/б изображению.
- `--output`: Путь для сохранения (по умолчанию `colored_image.jpg`).
- `--saturation`: Насыщенность (0.5–2.0, по умолчанию 1.0).
- `--style`: Стиль (`neutral`, `modern`, `vintage`, по умолчанию `neutral`).

### В Google Colab
```python
!python src/colorize.py --input examples/sample_images/bw/sample1.jpg
```

## Особенности
- Поддержка любых размеров изображений без размытия.
- Исправление пятен и искажений.
- Планируется добавление GAN для качества и постобработки.

Репозиторий: [https://github.com/T1desLuck/TintoraAi](https://github.com/T1desLuck/TintoraAi)