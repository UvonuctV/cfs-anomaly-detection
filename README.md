# Виявлення аномалій у кіберфізичних системах (CPS)

Вихідний код для магістерської дисертації: реалізація мультиваріативного конвеєра виявлення аномалій з використанням Stacking Regressor та методу Peaks-Over-Threshold (POT).

## Структура проекту

*   **`pipeline.py`** — Ядро конвеєра (генерація ознак, ансамблеве навчання, динамічний поріг POT, глобальний консенсус).
*   **`optimization.py`** — Модулі відбору ознак (Mutual Information, Random Forest Importance).
*   **`run_all.ipynb`** — Головний Jupyter-зошит для повного виконання, оцінки метрик (Precision/Recall/F1) та EDA візуалізацій.
*   **`run_batadal.py`, `run_hai.py`, `run_wadi.py`, `run_sherlock.py`** — Скрипти для автономного запуску через термінал на конкретних наборах даних.
*   **`draw_architecture.py`** — Скрипт для генерації архітектурних схем.

## Набори даних (Datasets)
Через обмеження розміру файлів на GitHub, самі дані не включені в репозиторій. Для запуску скриптів необхідно завантажити наступні публічні еталонні індустріальні датасети та розмістити їх у відповідних папках (наприклад, `../BATADAL/`):
1. **BATADAL** (Water Distribution Network)
2. **WADI** (Water Distribution Testbed)
3. **HAI** (HIL-based Augmented ICS)
4. **Sherlock IoT** (Smart grid/IoT environment)

## Швидкий запуск

```bash
# Встановлення залежностей
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn

# Запуск скрипта через термінал
python run_batadal.py

# Або інтерактивний запуск (для перегляду графіків і таблиць)
jupyter notebook run_all.ipynb
```
