[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Метеопрогноз для Сегеда: предсказание температуры

**Студент:** Климина Елена Андреевна

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия — предсказание непрерывного значения температуры воздуха

**Датасет:** Weather in Szeged 2006-2016 (Kaggle) — 96,453 записей с метеопараметрами

**Целевая метрика:** MAE (Mean Absolute Error) — наиболее интерпретируема для нас (в градусах Цельсия)


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed
│   │   ├── processed_weather.csv      # Финальные данные (95,912 строк, 257 признаков)
│   │   └── model_results.csv          # Таблица сравнения всех моделей
│   └── raw
│       └── weatherHistory.csv         # Исходный датасет (96,453 строк)
├── models
│   ├── final_gb_model.pkl             # Финальная Gradient Boosting модель
│   └── scaler.pkl                     # StandardScaler для нормализации
├── notebooks
│   
├── presentation
│   ├── eda_temperature.png            # EDA графики
│   ├── pca_analysis.png               # Анализ PCA
│   ├── model_comparison.png           # Сравнение моделей
│   ├── feature_importance.png         # Важность признаков
│   └── predictions_analysis.png       # Анализ предсказаний
├── report
│   └── report.md                      # Финальный отчёт
├── src
│   ├── preprocessing.py               # Предобработка данных (очистка, FE, сплит)
│   └── modeling.py                    # Обучение и оценка моделей
├── tests
│   └── test.py                        # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd hseml-group-project-vdrvvi-2

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить предобработку данных
python src/preprocessing.py

# 5. Запустить обучение моделей
python src/modeling.py
```

## Данные
### Исходные данные (`data/raw/`)
- **Источник:** Kaggle "Weather in Szeged 2006-2016"
- **Объём:** 96,453 строк, 12 столбцов
- **Признаки:** Formatted Date, Summary, Precip Type, Temperature (C), Apparent Temperature (C), Humidity, Wind Speed, Wind Bearing, Visibility, Loud Cover, Pressure, Daily Summary

### Предобработка (`data/processed/`)
- Удалён `Apparent Temperature (C)` — предотвращение дата-лика
- Удалены пропуски (517) и дубликаты (24)
- Удалены выбросы по температуре (-30°C до 45°C)
- Добавлены временные признаки (Hour, Month, DayOfYear, циклическое кодирование)
- Добавлены лаговые признаки (Temp_lag_1h, 3h, 6h)
- One-hot кодирование категориальных признаков
- **Итог:** 95,912 строк, 257 признаков

### Разделение данных (временное, без дата-лика)

| Set | Размер | Доля |
|-----|--------|------|
| Train | 67,132 | 70% |
| Validation | 14,380 | 15% |
| Test | 14,382 | 15% |


## Результаты
### Сравнение моделей (на валидации)

| Модель | MAE (°C) | RMSE (°C) | R² |
|--------|----------|-----------|-----|
| **Gradient Boosting (optimized)** | **0.568** | **0.757** | **0.9911** |
| Random Forest (n=200) | 0.593 | 0.803 | 0.9900 |
| Voting Ensemble (RF+GB) | 0.596 | 0.799 | 0.9901 |
| Gradient Boosting (n=200) | 0.623 | 0.824 | 0.9895 |
| Linear Regression | 0.713 | 0.930 | 0.9866 |
| KNN (k=10) | 2.815 | 4.017 | 0.7507 |

### Финальная модель на тестовых данных

| Метрика | Значение |
|---------|----------|
| **Test MAE** | **0.526°C** |
| Test RMSE | 0.717°C |
| Test R² | 0.9940 |

**Улучшение относительно baseline (среднее значение): 93.3%**

### Важность признаков (топ-5)

| Признак | Важность |
|---------|----------|
| Temp_lag_1h | 98.03% |
| Precip Type_snow | 0.44% |
| Hour_sin | 0.42% |
| Hour | 0.35% |
| Hour_cos | 0.17% |

### Выводы

1. **Лучшая модель:** Gradient Boosting с оптимизированными гиперпараметрами (`learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.8`)
2. **Наиболее важный признак:** Температура за предыдущий час (98% важности)
3. **PCA ухудшает качество:** Уменьшение размерности приводит к росту MAE с 0.57°C до 1.54°C
4. **Линейные модели работают хорошо** (MAE ~0.71°C), но уступают бустингам
5. **KNN показывает худшие результаты** (MAE ~2.8°C) из-за высокой размерности

### Сохранённые артефакты

| Файл | Описание |
|------|----------|
| `models/final_gb_model.pkl` | Финальная модель для инференса |
| `models/scaler.pkl` | StandardScaler для нормализации |
| `presentation/feature_importance.png` | Топ-20 наиболее важных признаков |
| `presentation/model_comparison.png` | Сравнение всех моделей (MAE и R²) |
| `presentation/pca_analysis.png` | Анализ PCA: объяснённая дисперсия |
| `presentation/predictions_analysis.png` | Анализ предсказаний и ошибок модели |



## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
