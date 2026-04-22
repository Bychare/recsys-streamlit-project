# recsys-streamlit-project

Streamlit-приложение с рекомендательными сценариями на датасете MovieLens `latest-small`.

Проект начинается с простых baseline-подходов и подготовлен к постепенному развитию: отдельная логика загрузки данных, препроцессинг, offline-оценка, тесты и Streamlit-интерфейс.

Проект сфокусирован на классическом UI рекомендательной системы и не содержит чат-интерфейса.

## Возможности

- автоматическая загрузка MovieLens `latest-small` при первом запуске;
- рекомендации по популярности;
- top-rated рекомендации с минимальным порогом числа оценок;
- поиск похожих фильмов через content-based cosine similarity;
- персональные рекомендации по `userId`;
- item-item collaborative filtering на user-item матрице;
- страница с обзором датасета;
- страница с offline-оценкой popularity baseline, top-rated baseline и item-item collaborative filtering;
- сравнение нескольких рекомендателей в UI;
- графики активности пользователей, жанров и связи рейтинга с популярностью фильмов;
- кривые `hit_rate@k`, `precision@k`, `coverage@k`, `novelty@k`;
- подготовка TF-IDF артефактов для content-based подхода;
- кеширование TF-IDF и item-user матриц в Streamlit-интерфейсе;
- Docker-окружение для локального запуска;
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

## Запуск в Docker

```bash
docker build -t recsys-streamlit .
docker run --rm -p 8501:8501 recsys-streamlit
```

В контейнере MovieLens также скачивается при первом запуске, если данных еще нет.

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
│   ├── collaborative.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── recommend.py
├── tests/
├── .dockerignore
├── Dockerfile
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

Команда считает `hit_rate@10`, `precision@10`, `coverage@10` и `novelty@10` для `popularity_baseline`, `top_rated_baseline` и `item_item_cf`, затем сохраняет результат:

```text
models/metrics.json
models/metrics_by_k.csv
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

- добавить SVD/ALS или гибридный content+collaborative подход;
- использовать `tags.csv` и `links.csv` для более богатых признаков фильмов;
- добавить PySpark/Airflow как отдельные инженерные компоненты после стабилизации MVP.
