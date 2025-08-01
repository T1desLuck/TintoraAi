# TintoraAI: Нейронная сеть для раскраски изображений

**TintoraAI** — это проект для автоматической раскраски чёрно-белых или деградированных изображений с использованием нейронной сети на основе архитектуры UNet и классификатора объектов. Проект поддерживает обучение на датасете с изображениями любых размеров и использует метки, сгенерированные нейронной сетью [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git).

## Основные возможности
- Раскраска чёрно-белых изображений с учётом семантической информации.
- Поддержка изображений любых размеров (с автоматическим паддингом).
- Оптимизированное обучение для GPU с 16 ГБ памяти.
- Конфигурация через `config.yaml` для удобной настройки.
- Запуск на локальном компьютере, Google Colab или Vast.ai.

## Структура проекта
```
TintoraAI/
├── tintora_dataset/        # Датасет
│   ├── bw/                 # Чёрно-белые изображения (.jpg)
│   ├── color/              # Цветные изображения (.jpg)
│   └── labels/             # Метки (.npy, созданы NpyLabelNet)
├── src/                    # Исходный код
│   ├── model/              # Модели и обработка изображений
│   └── training/           # Датасет и обучение
├── tests/                  # Тесты
├── .github/workflows/      # CI/CD
├── config.yaml             # Конфигурация
├── requirements.txt        # Зависимости
├── README.md               # Описание проекта
├── INSTALL.md              # Инструкции по установке
├── TRAINING.md             # Инструкции по обучению
├── USAGE.md                # Инструкции по использованию
└── CONTRIBUTING.md         # Инструкции для контрибьюторов
```

## Быстрый старт
1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/<your-username>/TintoraAI.git
   cd TintoraAI
   ```
2. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Настройте датасет**:
   - Поместите датасет в папку `tintora_dataset/` с подпапками `bw/`, `color/`, `labels/`.
   - Метки создаются с помощью [NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet.git).
4. **Настройте `config.yaml`**:
   - Укажите путь к датасету в `data_path` (например, `./tintora_dataset`).
5. **Запустите обучение**:
   ```bash
   python src/training/train.py --config config.yaml
   ```
6. **Раскрасьте изображение**:
   ```bash
   python src/colorize.py --input path/to/image.jpg --output colored_image.jpg
   ```

Подробные инструкции в `INSTALL.md`, `TRAINING.md` и `USAGE.md`.

## Лицензия
Пока не определена. Свяжитесь с автором для уточнения.

## Контакты
- GitHub: [T1desLuck](https://github.com/T1desLuck)
- Вопросы и предложения: открывайте issue в репозитории.