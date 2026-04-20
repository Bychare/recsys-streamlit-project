# recsys-streamlit-project

Streamlit-приложение с рекомендательными сценариями на датасете MovieLens `latest-small`.

Проект начинается с простых baseline-подходов и подготовлен к постепенному развитию: отдельная логика загрузки данных, препроцессинг, offline-оценка, тесты и Streamlit-интерфейс.

## Возможности

- автоматическая загрузка MovieLens `latest-small` при первом запуске;
- рекомендации по популярности;
- top-rated рекомендации с минимальным порогом числа оценок;
- поиск похожих фильмов через content-based cosine similarity;
- страница с обзором датасета;
- страница с offline-оценкой popularity baseline;
- подготовка TF-IDF артефактов для content-based подхода;
- тесты с coverage-отчетом.

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/Home.py
```

После запуска приложение будет доступно по адресу:

```text
http://localhost:8501
```

При первом запуске MovieLens скачивается в `data/raw/`.

Не запускайте `app/Home.py` как обычный Python-файл. Для Streamlit нужен запуск через `streamlit run app/Home.py`.

## Структура

```text
.
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 01_Data_Overview.py
│       └── 02_Model_Evaluation.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── scripts/
│   ├── build_artifacts.py
│   └── evaluate.py
├── src/
│   ├── analytics.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── recommend.py
├── tests/
├── pytest.ini
├── .gitignore
├── README.md
└── requirements.txt
```

## Подготовка артефактов

Артефакты не хранятся в Git и создаются локально:

```bash
python scripts/build_artifacts.py
```

Команда создает:

```text
data/processed/movies_features.parquet
data/processed/rating_stats.parquet
models/tfidf_vectorizer.joblib
models/tfidf_matrix.npz
```

## Offline-оценка

```bash
python scripts/evaluate.py
```

Команда считает `hit_rate@10` и `precision@10` для popularity baseline и сохраняет результат:

```text
models/metrics.json
```

Посмотреть сохраненный JSON:

```bash
cat models/metrics.json
```

## Проверки

```bash
pytest
```

В конце `pytest` показывает coverage-отчет. Проценты рядом с файлами тестов во время выполнения, например `[36%]`, означают только прогресс запуска тестов.

## Данные и артефакты в Git

В репозиторий добавляются только `.gitkeep`-файлы для структуры каталогов. Сырые данные, подготовленные таблицы и модельные артефакты игнорируются через `.gitignore`, потому что они воспроизводятся командами выше.

## Следующие улучшения

- добавить персональные рекомендации по `userId`;
- добавить item-item collaborative filtering;
- сравнить несколько рекомендателей в UI;
- подготовить Docker-окружение для запуска на сервере;
- добавить PySpark/Airflow как отдельные инженерные компоненты после стабилизации MVP.
