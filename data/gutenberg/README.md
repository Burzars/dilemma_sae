# Кластеризация книг Project Gutenberg по метаданным

Берёт датасет Kaggle [`lokeshparab/gutenberg-books-and-metadata-2025`](https://www.kaggle.com/datasets/lokeshparab/gutenberg-books-and-metadata-2025)
(~75k книг + метаданные) и выдаёт тематические кластеры:

```json
[
  {"cluster_name": "Detective Fiction", "ids": [1234, 5678, ...]},
  {"cluster_name": "Cookbooks",         "ids": [...]},
  ...
]
```

**Метод:** эмбеддинги метаданных (Title + LCSH-темы + полки Gutenberg) →
кластеризация (KMeans) → имена кластеров от LLM. Жёсткое разбиение — каждая
книга ровно в одном кластере.

## Установка

```bash
pip install -r data/gutenberg/requirements.txt
# torch/transformers/pandas/numpy/pyyaml/tqdm/openai уже в ../../requirements.txt
```

## Запуск (на сервере)

```bash
export OPENROUTER_API_KEY=sk-...          # для LLM-имён (тот же ключ, что в проекте)
cd data/gutenberg
python cluster_books.py                   # качает датасет с Kaggle, GPU подхватится авто
# → outputs/clusters.json
```

Датасет уже скачан? Укажите путь (папку или CSV):
```bash
python cluster_books.py --data-dir /path/to/gutenberg-metadata
```

Без LLM (имена из характерных ключевых слов/полок, офлайн, бесплатно):
```bash
python cluster_books.py --no-llm
```

## Стадии и кэш

Пайплайн из 4 стадий; каждая пишет артефакт в `outputs/` и **пропускается, если
он уже есть** (как `scripts/01_extract.py`). Эмбеддинг — самое дорогое, поэтому
смена числа кластеров или переименование не требуют пересчёта эмбеддингов.

| стадия  | артефакт                    | что делает                                   |
|---------|-----------------------------|----------------------------------------------|
| load    | `books.pkl`                 | CSV → детект колонок → парс subjects/LoCC/полок |
| embed   | `embeddings.npy`            | sentence-transformers по doc_text            |
| cluster | `assignments.csv`           | (опц. UMAP) + KMeans/HDBSCAN                  |
| name    | `clusters.json` (+ detailed, summary.csv) | c-TF-IDF + LLM-имена           |

```bash
python cluster_books.py --from cluster            # пересчитать только cluster+name
python cluster_books.py --from name --set cluster.k=120   # ← сначала смените k и --from cluster
python cluster_books.py --force                   # пересчитать всё
```

## Конфиг

Всё в `config.yaml`; переопределение из CLI через `--set`:

```bash
python cluster_books.py --set cluster.k=120 embed.model=BAAI/bge-base-en-v1.5 data.languages=null
```

Ключевые параметры: `cluster.k` (гранулярность), `cluster.algo` (kmeans|hdbscan),
`embed.model`, `data.languages` (`["en"]` по умолчанию; `null` = все языки),
`name.use_llm`.

## Проверка без GPU и без датасета

`--fake-embed` заменяет модель детерминированными хеш-эмбеддингами (без torch):
```bash
python cluster_books.py --data-dir <папка с CSV> --smoke --fake-embed
```

## Выход (`outputs/`)

- `clusters.json` — итог: `[{cluster_name, ids}]`, кластеры по убыванию размера.
- `clusters_detailed.json` — размеры, топ-полки/темы/ключевые слова, примеры заголовков.
- `clusters_summary.csv` — то же в табличном виде для быстрой глазной проверки имён.

`outputs/` в `.gitignore` (крупные артефакты).

## Доступ к тексту книги по id

`clusters.json` содержит списки Gutenberg id; сам текст достаётся утилитой
[`book_text.py`](book_text.py). Раскладка датасета определяется **автоматически** —
тексты могут лежать отдельными файлами (`{id}.txt` / `pg{id}.txt`) **или** колонкой
`text`/`context` в CSV/parquet; плюс всегда есть фолбэк-загрузка с gutenberg.org по id.

```bash
# точно узнать, КАК хранятся тексты в датасете (запустить на сервере):
python book_text.py --inspect --data-dir /path/to/dataset

# текст одной книги в stdout (с вырезанной шапкой/подвалом Gutenberg):
python book_text.py --id 1342 --strip-headers

# выгрузить все тексты кластера в outputs/texts/:
python book_text.py --cluster "Detective Fiction" --strip-headers
```

`--data-dir` можно не указывать — берётся из `config.yaml` (или качается с Kaggle).
Скачанные с gutenberg.org тексты кэшируются в `outputs/text_cache/`.
