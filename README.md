# dilemma_sae

Пайплайн интерпретации hidden states LLM через Sparse Autoencoder на
задаче Volunteer's Dilemma. Рефакторинг исходного монолитного ноутбука
в модульный проект для запуска на сервере.

## Структура

```
dilemma_sae/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # все параметры пайплайна
├── src/
│   ├── data.py               # ReasoningSample, load_samples
│   ├── extract.py            # _build_chat_text, extract_activations, ActivationConfig
│   ├── sae.py                # STEHeaviside, SparseAutoencoder, train_sae
│   ├── analysis.py           # encode_all, neuron_statistics, top_contexts,
│   │                         # label_contrast, sample_level_contrast
│   ├── interpret.py          # LLM-судья (OpenRouter) → темы нейронов
│   ├── visualize.py          # графики (train curves, Pareto, fire_rate hist)
│   └── io_utils.py           # save/load активаций, SAE, конфиги, логирование
├── scripts/
│   ├── 01_extract.py         # экстракция hidden states из LLM
│   ├── 02_train_sae.py       # обучение одной SAE-архитектуры
│   ├── 03_analyze.py         # encode_all + stats + top + контраст + отчёт
│   ├── 04_interpret.py       # LLM-судья → темы → summary.csv
│   ├── 05_visualize.py       # PNG-графики
│   └── _common.py            # общий парсер CLI
├── notebook.ipynb            # sandbox для интерактивного просмотра нейронов
└── artifacts/                # сюда пишутся все артефакты (создаётся автоматически)
    ├── logs/
    └── figures/
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если нужна 4-bit квантизация LLM при экстракции — установите
`bitsandbytes` отдельно. Намеренно не включён в requirements: в Colab
он часто конфликтует с triton.

## Данные

В `data/results_dilemma/` должны лежать 8 JSON-файлов
`volunteer_data_llama_reasoning_<N>_res.json`. Путь меняется в
`configs/default.yaml`, поле `data.results_dir`.

Формат одного файла:
```json
{
  "prompt": "A neighborhood has lost its electrical power...",
  "answers": [
    {"answer": "In a hypothetical scenario...", "type": "Yes"},
    ...
  ]
}
```

## Запуск пайплайна

Полный прогон с нуля:

```bash
# Шаг 1: извлечение активаций (10-20 минут на T4)
python scripts/01_extract.py

# Шаг 2: обучение SAE. По умолчанию topk.
python scripts/02_train_sae.py
# Чтобы обучить все три варианта:
for mode in relu_l1 topk jumprelu; do
    python scripts/02_train_sae.py --override sae.mode=$mode
done

# Шаг 3: анализ (encode_all, статистика, top нейронов, Yes/No контраст, отчёт)
python scripts/03_analyze.py

# Шаг 4: LLM-судья (нужен OPENROUTER_API_KEY)
export OPENROUTER_API_KEY=sk-...
python scripts/04_interpret.py

# Шаг 5: PNG-графики в artifacts/figures/
python scripts/05_visualize.py

# Шаг 6 (опц.): held-out EV (обобщает ли SAE). GPU, ~25 мин на 8 фолдов.
python scripts/06_eval_ev.py --smoke    # сначала дёшево проверить
python scripts/06_eval_ev.py

# Шаг 7 (опц., дорого): steering-эксперимент (причинность триггерных нейронов).
# Нужен OPENROUTER_API_KEY (судья Yes/No/?). Сначала ОБЯЗАТЕЛЬНО --smoke.
export OPENROUTER_API_KEY=sk-...
python scripts/07_steering.py --smoke   # 4 генерации — проверка пайплайна
python scripts/07_steering.py           # полный прогон, ~3 ч
```

Шаги 6 и 7 — отдельные эксперименты поверх обученного SAE, не часть
основного прогона. Оба переиспользуют `parse_args_and_init`, `--config`,
`--override`, `--force` и кешируют результат (07 — пояячейково, по
`neuron × alpha`, с резюмированием). У обоих есть `--smoke` для безопасной
дешёвой проверки на ноутбуке/малом бюджете.

**Steering и индексы нейронов.** Индексы признаков SAE привязаны к
конкретному обученному `sae_<mode>.pt`. После переобучения SAE старые ID
невалидны, поэтому `07_steering.py` по умолчанию выбирает цели
ПРОГРАММНО из Yes/No-контраста текущего SAE (+ один случайный контроль).
Явный список можно задать через `steering.neuron_ids` в конфиге — но
только если приложен «родной» SAE, породивший эти ID.

### Переопределение параметров

Все скрипты принимают `--config` и `--override`:

```bash
python scripts/02_train_sae.py \
    --config configs/default.yaml \
    --override sae.mode=topk sae.k=16
```

`--override` поддерживает dotted-нотацию, можно задавать несколько
переопределений подряд. Значения парсятся в типы (int / float / bool / str).

### Кеширование

Каждый шаг проверяет, есть ли его артефакт на диске, и пропускает себя,
если есть. Чтобы пересчитать принудительно — `--force`. Например:

```bash
python scripts/02_train_sae.py --override sae.mode=topk sae.k=32 --force
```

## Артефакты

Все пишутся в `./artifacts/` (путь задаётся `paths.artifacts_dir` в
YAML). Список:

| Файл                              | Создаёт         | Содержимое                              |
|-----------------------------------|-----------------|-----------------------------------------|
| `activations.npz`                 | 01_extract      | hidden / sample_idx / token_pos / label |
| `samples.json`                    | 01_extract      | сериализованные ReasoningSample         |
| `sae_<mode>.pt`                   | 02_train_sae    | state_dict + history + config           |
| `features_<mode>.npz`             | 03_analyze      | матрица features (encode_all)           |
| `neuron_stats_<mode>.npz`         | 03_analyze      | fire_rate / mean_active / max_active /  |
|                                   |                 | n_fires                                 |
| `neuron_report_<mode>.json`       | 03_analyze      | top контексты + Yes/No триггеры         |
| `neuron_themes_<mode>.json`       | 04_interpret    | то же + поле `theme` от LLM             |
| `summary_<mode>.csv`              | 04_interpret    | компактная сводка                       |
| `figures/*.png`                   | 05_visualize    | графики                                 |
| `eval_ev.json`                    | 06_eval_ev      | train/test EV+MSE по фолдам и random    |
| `figures/ev_per_file.png`         | 06_eval_ev      | bar chart train vs test EV по файлам    |
| `steering_results.json`           | 07_steering     | counts+answers по neuron × alpha × prompt |
| `steering_summary.csv`            | 07_steering     | neuron/theme/alpha → yes/no/q, p_yes    |
| `figures/steering_curves.png`     | 07_steering     | p(Yes) vs alpha по нейронам             |
| `logs/<script_name>.log`          | каждый скрипт   | дублирует stdout                        |

`steering_results.json` может вырасти до нескольких МБ (хранит тексты
генераций) — при необходимости добавьте его в `.gitignore`.

## Миграция существующих артефактов

Если у вас уже есть `activations.npz`, `sae_topk.pt` и т.п. с прошлого
запуска (например, с Google Drive), просто скопируйте их в `./artifacts/`:

```bash
mkdir -p artifacts
cp /path/to/old/activations.npz   artifacts/
cp /path/to/old/samples.json      artifacts/
cp /path/to/old/sae_topk.pt       artifacts/
cp /path/to/old/sae_jumprelu.pt   artifacts/
```

Старый файл `sae.pt` (без суффикса режима, из самых ранних экспериментов
с relu_l1) — переименуйте в `sae_relu_l1.pt`, если хотите его
загружать:

```bash
cp /path/to/old/sae.pt artifacts/sae_relu_l1.pt
```

После этого `01_extract` / `02_train_sae` пропустят свои шаги (артефакты
уже на месте), а `03_analyze` / `04_interpret` / `05_visualize`
заработают сразу.

`neuron_report.json` из старых прогонов **не мигрируйте** — его формат
изменился (поля `kind`, дополнительная статистика), пересоберите
через `03_analyze.py`.

## Запуск на сервере (без дисплея)

Все графики (`05_visualize.py`) используют backend `matplotlib.Agg` и
сохраняют PNG в `artifacts/figures/`. Никакого `plt.show()` нет, можно
скачивать готовые PNG.

Для длительных прогонов на сервере удобно через `nohup`:

```bash
nohup python scripts/01_extract.py > /dev/null 2>&1 &
tail -f artifacts/logs/01_extract.log
```

## Ноутбук

`notebook.ipynb` — короткий sandbox. Не делает обучения и экстракции,
только грузит готовые артефакты с диска и позволяет интерактивно
полистать нейроны (поменять `NEURON_ID` и перезапустить ячейку).
Используется на локальной машине после прогона пайплайна на сервере
(скачать `artifacts/` себе и запустить ноутбук).

## Известные проблемы

См. `HANDOVER.md` (отдельный файл). Кратко:

1. **Топ-15 контекстов первых ~5 нейронов могут совпадать.** Это
   следствие дедупликации по `sample_idx` в `top_contexts_for_neuron`
   — она показывает только первый хит на каждый пример. В отчёт
   добавлено поле `token_pos`, чтобы можно было различать; полный фикс
   потребует развернуть `top_contexts_for_neuron` на два режима
   (с дедупликацией / без).
2. **JumpReLU STE исправлен** (`src/sae.py`: `JumpReLUFunction` /
   `HeavisideFunction` по Rajamanoharan et al. 2024 — отдельные псевдо-
   градиенты по порогу, без паразитного члена к `W_enc`). Главная причина
   прошлых плохих результатов — слишком малые `theta_init=0.1`/`ste_eps=0.1`
   при масштабе пре-активаций ~единицы-десятки: STE-окно почти всегда пустое,
   и θ не обучался. Дефолты подняты до `theta_init=1.0`, `ste_eps=1.0`
   (разумный диапазон θ 1–3, ε 0.5–2; `l0_coef` при необходимости поднять).
   `topk` и `relu_l1` не затронуты.
3. **Дисбаланс Yes/No 547:32.** Делает No-сторону контраста шумной.
   Альтернатива через config:
   `--override contrast.neg_label="?"` — статистически чище (61 пример).

## Принципы рефакторинга

Эта итерация — РЕФАКТОРИНГ, не доработка. Логика и формулы лоссов SAE,
трюк с удалением параллельной составляющей градиента `W_dec`, дважды
вычисляемый контраст (token + sample), кеширование артефактов — всё
сохранено как было. Любые замеченные баги отмечены в коде комментариями
и здесь — но не правились.
