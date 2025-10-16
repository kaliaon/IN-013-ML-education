# Проект: Аналитика образовательных данных (практическая часть диссертации)

## Быстрый старт

1. Создать и активировать виртуальное окружение.
2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустить Streamlit-дэшборд:
   ```bash
   streamlit run src/visualization/dashboard.py
   ```

## Эксперименты

- Ноутбуки: `notebooks/EDA.ipynb`, `notebooks/Experiments.ipynb`
- Модули:
  - Загрузка/очистка: `src/data/`
  - Кластеризация: `src/models/clustering.py`
  - Прогнозирование: `src/models/prediction.py`
  - Метрики/сравнение: `src/evaluation/`
  - Фичи/SHAP: `src/features/`

## Структура

- `src/data`: загрузка, очистка, нормализация, разбиение
- `src/models`: кластеризация, прогнозирование
- `src/evaluation`: метрики и сравнение
- `src/visualization`: графики и Streamlit
- `notebooks`: EDA и эксперименты
- `data/raw/oulad`: сюда поместите распакованные таблицы OULAD (`*.csv`)
- `data/processed`: сохранённые подготовленные датасеты

## Следующие шаги

- Добавить ноутбуки EDA/экспериментов
- Добавить SHAP-анализ и сравнительную таблицу моделей


