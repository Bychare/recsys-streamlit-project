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
- персональные рекомендации через SVD-факторизацию user-item матрицы;
- гибридные рекомендации на основе CF, SVD и популярности;
- страница с обзором датасета;
- страница с offline-оценкой popularity baseline, top-rated baseline, item-item collaborative filtering, SVD и hybrid;
- сравнение нескольких рекомендателей в UI;
- графики активности пользователей, жанров и связи рейтинга с популярностью фильмов;
- кривые `hit_rate@k`, `precision@k`, `coverage@k`, `novelty@k`;
- таблицы и кривые с `ndcg@k` для сравнения качества ранжирования;
- подготовка TF-IDF артефактов для content-based подхода;
- кеширование TF-IDF и item-user матриц в Streamlit-интерфейсе;
- Docker-окружение и `docker compose` для локального и серверного запуска;
- `Makefile` с типовыми командами разработки;
- GitHub Actions CI для тестов и проверки сборки Docker-образа;
- тесты с coverage-отчетом.

## Быстрый старт

```bash
make install
make run
```

После запуска приложение будет доступно по адресу:

```text
http://localhost:8501
```

При первом запуске MovieLens скачивается в `data/raw/`.

Не запускайте `app/Home.py` как обычный Python-файл. Для Streamlit нужен запуск через `streamlit run app/Home.py`.

Если удобнее вручную, локальный запуск без `make` по-прежнему выглядит так:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/Home.py
```

## Запуск в Docker

```bash
make docker-build
make docker-run
```

В контейнере MovieLens также скачивается при первом запуске, если данных еще нет.

## Запуск через Docker Compose

По умолчанию compose пробрасывает `8501:8501`. Если нужен другой порт, можно создать локальный `.env` на основе `.env.example`.

```bash
make compose-up
make compose-logs
make compose-down
```

Compose сохраняет скачанные данные и модельные артефакты в локальные каталоги:

```text
data/raw/
data/processed/
models/
```

## Структура

```text
.
.github/
│   └── workflows/
│       └── ci.yml
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
│   ├── hybrid.py
│   ├── matrix_factorization.py
│   ├── preprocess.py
│   └── recommend.py
├── tests/
├── .env.example
├── .dockerignore
├── Dockerfile
├── Makefile
├── docker-compose.yml
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

Команда считает `hit_rate@10`, `precision@10`, `ndcg@10`, `coverage@10` и `novelty@10` для `popularity_baseline`, `top_rated_baseline`, `item_item_cf`, `svd_recommender` и `hybrid_recommender`, затем сохраняет результат:

```text
models/metrics.json
models/metrics_by_k.csv
```

Посмотреть сохраненный JSON:

```bash
cat models/metrics.json
```

`recall@k` отдельно не выводится: в текущем `leave-one-out` сценарии у каждого пользователя в test только один релевантный фильм, поэтому `recall@k` здесь совпадает с `hit_rate@k`.

## Проверки

```bash
make test
```

В конце `pytest` показывает coverage-отчет. Проценты рядом с файлами тестов во время выполнения, например `[36%]`, означают только прогресс запуска тестов.

Полезные команды разработки:

```bash
make artifacts
make evaluate
make ci
```

`make ci` локально повторяет базовую проверку из GitHub Actions: `compileall + pytest`.

## CI

В `.github/workflows/ci.yml` настроены два джоба:

- запуск тестов на Python 3.12;
- проверка сборки Docker-образа.

## Данные и артефакты в Git

В репозиторий добавляются только `.gitkeep`-файлы для структуры каталогов. Сырые данные, подготовленные таблицы и модельные артефакты игнорируются через `.gitignore`, потому что они воспроизводятся командами выше.

## Следующие улучшения

- добавить ALS или Spark-версию рекомендателя для более крупного датасета;
- подготовить отдельные конфиги деплоя под reverse proxy и systemd;
- использовать `tags.csv` и `links.csv` для более богатых признаков фильмов;
- добавить PySpark/Airflow как отдельные инженерные компоненты после стабилизации MVP.
