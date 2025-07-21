# Установка TintoraAI

Этот документ описывает, как установить и настроить TintoraAI для локального или облачного запуска на CPU или GPU.

## Требования
- **Операционная система**: Windows, macOS, Linux.
- **Python**: 3.8 или новее.
- **Аппаратное обеспечение**:
  - CPU: Любой современный процессор (рекомендуется 4+ ядра для скорости).
  - GPU: Опционально (NVIDIA с CUDA для ускорения).
  - ОЗУ: Минимум 8 ГБ (16 ГБ для обучения).
- **Интернет**: Для установки зависимостей и работы в облаке.

## Установка локально
1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/your-username/TintoraAI.git
   cd TintoraAI
   ```

2. **Создайте виртуальное окружение** (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```
   Зависимости включают `torch`, `torchvision`, `streamlit`, `fastapi`, `uvicorn`, `pillow`, `pytest`, `flake8`.

4. **Проверка установки**:
   Запустите веб-интерфейс:
   ```bash
   streamlit run src/ui/app.py
   ```
   Откройте браузер по адресу `http://localhost:8501`.

## Установка в Google Colab
1. Откройте [Google Colab](https://colab.research.google.com).
2. Загрузите репозиторий:
   ```bash
   !git clone https://github.com/your-username/TintoraAI.git
   %cd TintoraAI
   ```
3. Установите зависимости:
   ```bash
   !pip install -r requirements.txt
   ```
4. Установите ngrok для доступа к интерфейсу:
   ```bash
   !pip install pyngrok
   !ngrok authtoken YOUR_NGROK_TOKEN
   ```
   Замените `YOUR_NGROK_TOKEN` на ваш токен (получите на [ngrok.com](https://ngrok.com)).
5. Запустите Streamlit:
   ```bash
   !streamlit run src/ui/app.py &>/dev/null &
   !ngrok http 8501
   ```
   Перейдите по ссылке от ngrok для доступа к интерфейсу.

## Облачное развертывание
### Streamlit Cloud
1. Создайте аккаунт на [Streamlit Cloud](https://streamlit.io/cloud).
2. Свяжите ваш GitHub-репозиторий с TintoraAI.
3. Укажите `src/ui/app.py` как основной файл.
4. Streamlit Cloud автоматически установит зависимости из `requirements.txt`.

### Render
1. Создайте аккаунт на [Render](https://render.com).
2. Создайте новый Web Service, подключив ваш GitHub-репозиторий.
3. Настройте:
   - Build Command: `pip install -r requirements.txt`.
   - Start Command: `streamlit run src/ui/app.py --server.port $PORT` (для UI) или `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT` (для API).
4. Разверните сервис.

## Устранение неполадок
- **Ошибка CUDA**: Если GPU недоступен, TintoraAI автоматически использует CPU. Убедитесь, что установлена версия PyTorch с поддержкой CUDA (`pip install torch torchvision`).
- **Зависимости**: Если установка зависает, обновите pip: `pip install --upgrade pip`.
- **Проблемы с ngrok**: Проверьте токен и лимиты бесплатного плана.