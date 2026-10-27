# Phase 2A — результаты TF-IDF detector

2026-10-22. QA RAGTruth, snapshot `c103204b9ce28d6bbad859304bf30de72b8ed8fe`,
label policy include/all. Это локальный benchmark, не сравнение с опубликованными методами.
Design зафиксирован в `PHASE2A_DESIGN.md` (historical local artifact; not distributed) до dev evaluation.
После просмотра dev ни score, ни preprocessing, ни threshold не изменялись.

## Данные и calibration

| Partition | Ответы | Source groups | Class 0 / 1 |
|---|---:|---:|---:|
| Train | 3216 | 536 | 2221 / 995 |
| Calibration | 810 | 135 | 548 / 262 |
| Dev | 1008 | 168 | 701 / 307 |
| Official test | 900 | 150 | withheld |

Phase 1.1 partitions сохранены. Все 1008 dev response IDs / 168 source IDs совпадают с
историческим Phase 1 reference. Все шесть source/response overlap checks нулевые.
Raw hashes прежние; test labels закрыты, test metrics не вычислялись.
TF-IDF fit: 536 уникальных train источников, 2046 chunks, vocabulary 71137 features.
Question/response/calibration/dev/test не входят в vocabulary/IDF fit.

Проверены 518 calibration boundaries: все distinct scores + all-negative boundary.
Prediction `risk >= threshold`, цель macro-F1, exact ties → highest threshold.
Выбран **0.9694931298302414** и записан в freeze и design **до** dev.
Следующие значения относятся исключительно к selection на calibration:

| Calibration metric | Значение |
|---|---:|
| Accuracy | 0.6333333333 |
| Macro-F1 | 0.6162294870 |
| Positive precision / recall / F1 | 0.453581 / 0.652672 / 0.535211 |
| AUROC / Average Precision | 0.675283 / 0.441843 |
| [[TN,FP],[FN,TP]] | [[342,206],[91,171]] |

Источник: `reports/lexical_calibration.json`; все candidate metrics сохранены.
Это оптимизированные selection metrics, не unbiased оценка.

## Frozen dev: метрики и uncertainty

Знаменатель: 1008 ответов, 701 class 0 / 307 class 1, 168 источников.
В скобках 95% percentile CI, 1000 paired source-level bootstrap resamples, seed42.

| Метрика | TF-IDF | Majority |
|---|---|---|
| Accuracy | 0.611111 [0.577381, 0.641890] | 0.695437 [0.657738, 0.730184] |
| Macro-F1 | 0.590909 [0.554529, 0.621704] | 0.410181 [0.396768, 0.422027] |
| Positive precision | 0.410901 [0.359496, 0.460398] | 0 [0, 0] |
| Positive recall | 0.638436 [0.567188, 0.699700] | 0 [0, 0] |
| Positive F1 | 0.500000 [0.450066, 0.542964] | 0 [0, 0] |
| AUROC | 0.665954 [0.626098, 0.700808] | 0.500000 [0.5, 0.5] |
| Average Precision | 0.417237 [0.364967, 0.470355] | 0.304563 [0.269816, 0.342262] |
| [[TN,FP],[FN,TP]] | [[420,281],[111,196]] | [[701,0],[307,0]] |

Majority positive precision не определён из-за отсутствия positive predictions: используется
заранее заданная zero-division convention 0, флаг сохранён. Ranking score majority — константа
995/3216. Старые majority classification metrics не изменились; historical CLI по-прежнему
не вычисляет ranking metrics. AP — sklearn Average Precision, не trapezoidal PR-AUC.

Paired Δmacro-F1 = **+0.180728**, CI **[+0.139899, +0.216048]**.
Δaccuracy = **−0.084325**, CI **[−0.133929, −0.037674]**.
ΔAUROC CI [+0.126098, +0.200808]; ΔAP CI [+0.077997, +0.147629].
Все 1000 draws дали определённые ranking metrics. CI условны на fixed model/threshold,
не включают fitting/calibration uncertainty и не исправляют repeated dev selection.
Confusion-count intervals тоже сохранены в `reports/lexical_dev.json`.

Вывод: score несёт сигнал и обнаруживает 196 из 307 positive answers, но даёт 281 false alarm.
Рост macro-F1 сопровождается падением accuracy. Самостоятельная надёжная проверка фактов
не получилась; это воспроизводимая отправная точка для семантического detector.

## Runtime и воспроизводимость

Windows 11, Python 3.12.14, CPU (12 logical CPUs; локальная машина i5-10400F), GPU не использован.
NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.6.0, threadpoolctl 3.7.0.

| Stage | Wall time |
|---|---:|
| TF-IDF fit | 0.524132 s |
| Calibration scoring (810) | 6.985312 s; 115.96 responses/s |
| Threshold search | 0.002602 s |
| Dev model load | 0.121852 s |
| Dev scoring (1008) | 7.850942 s; 128.39 responses/s |
| Dev metrics + bootstrap | 4.192530 s |
| Dev diagnostics/error artifact write | 0.179763 s |
| Dev stage before final report write | 12.498391 s |
| Majority fit | 0.000232 s |
| Majority predictions + constant scores (1008) | 0.0000064 s |

Scoring включает sentence/chunk diagnostics и context cache между ответами одного source.
Не включает raw loading/validation, bootstrap, disk write. Input preparation записана отдельно
в JSON. Это единичный wall-time замер, не latency benchmark с CI. Формальная majority throughput
около 157 млн responses/s сохранена в JSON, но микросекундный замер константного списка
зависит от timer resolution и не характеризует end-to-end pipeline.

Повторная оценка того же freeze: metrics и bootstrap полностью совпали; dev.jsonl и errors.json
побайтно одинаковы. В 1818 calibration+dev ответах проверены 16414 sentence offsets и
49242 retrieved chunk offsets: каждый точно ссылается на исходный ответ/evidence/source_id.
Доказательства: `reports/lexical_dev_reproduced.json`, `reports/phase2a_integrity.json`.

Локально: 105 passed с real-data opt-in (104 passed, 1 skipped без него), Ruff passed,
pip check passed. CI matrix 3.11/3.12 включает все offline regressions. Файл workflow
существовал, но был untracked; теперь `.github/workflows/ci.yml` добавлен в Git index.
Commit/push/remote CI run не выполнялись. Локальный Python 3.11 не проверен.

## Разбор 15 FP и 15 FN

Сохранены полные answer text, gold, prediction, score, question, original evidence,
gold spans и top-3 retrieval для каждого sentence в `predictions/phase2a/errors.json`.
Файл игнорируется Git. Выбор deterministic SHA256 order, сначала разные source IDs;
это качественная выборка, не случайная оценка частот причин во всех ошибках.
Ниже ручной разбор всех 30 выбранных примеров; source texts не перепечатываются в Git-документ.
Gold labels не исправлялись даже в неоднозначных случаях.

**FP: gold=0, prediction=1.** Основная причина условная: у одного ответа могут быть несколько.

| Response ID | Основное наблюдение |
|---|---|
| 12123 | Короткий fragment `(complement)` имеет support=0, хотя термин есть в evidence; max pooling реагирует на фрагмент. |
| 12579 | Пункт `Scissors` не сопоставляется с формой `scissor`; morphology/short-list mismatch. |
| 13116 | `incorrect_refusal` без hallucination spans; отказ имеет почти нулевую similarity. |
| 17768 | Отказ без hallucination spans ошибочно считается неподдержанным утверждением. |
| 16351 | Отказ по вопросу о dependent tax filing; detector не отделяет abstention от claims. |
| 17542 | Вводная фраза перед ответом об орхидеях становится самым рискованным sentence. |
| 11949 | Номер списка `3.` отделился в sentence и дал risk=1. |
| 15994 | Номер списка `4.` даёт risk=1. |
| 17081 | `antalgic gait` есть в evidence, но перефразированный пункт получает support≈0.0235 и неверный top-1. |
| 17495 | Завершающая фраза `That's it!` даёт почти максимальный risk. |
| 14119 | `incorrect_refusal` по рецепту, без hallucination spans; модель ошибочно сигналит. |
| 16532 | Отказ/оговорка о недостатке данных для стоимости жизни в Dubrovnik; gold=0. |
| 12253 | Отдельный номер списка `1.` даёт risk=1. |
| 13348 | Отдельный номер списка `1.` даёт risk=1. |
| 16703 | Отказ с пояснением о dermatologist; max pooling выделяет не факт, а abstention. |

В этой выборке: 6 refusal, 5 list/fragment, 2 boilerplate, 1 morphology, 1 paraphrase/retrieval.
Это обнаруженные ограничения сегментации и lexical features, а не доказательство, что все FP
объясняются только ими. Изменять сегментацию задним числом для текущей таблицы запрещено.

**FN: gold=1, prediction=0.** Высокий overlap может маскировать неподдержанное дополнение.

| Response ID | Основное наблюдение |
|---|---|
| 11898 | Дополнения о practical experience и доходах рядом с поддержанными фактами; часть gold boundaries требует осторожной интерпретации. |
| 11879 | К cream fat content добавлены serving suggestions (`implicit_true`), lexical score их не отделяет. |
| 14900 | Функция голосового органа перенесена на air sacs; topical overlap не проверяет отношение между сущностями. |
| 12646 | Неподтверждённые температура/месяцы/сезонность Shanghai; похожие weather terms маскируют новые детали. |
| 13503 | Перепутаны даты и время Bangkok/Chicago, добавлены 14 часов; числа в контексте не означают верную связь. |
| 13118 | Утверждение «не указано, как часто заниматься» противоречит источнику с частотой; negation token сам по себе не entailment. |
| 14656 | Значение BMI для детей перенесено на взрослых мужчин; число присутствует в evidence, но относится к другой группе. |
| 16534 | Из нескольких городских сравнений выведена общая доступность Dubrovnik; unsupported generalization с общими словами. |
| 14520 | `stress relief` помечен implicit_true рядом с massage benefit; тонкая граница между inference и явной поддержкой. |
| 17118 | Gold отмечает детали ice-pack timing и medical advice, хотя близкая формулировка есть в passage 3; возможная annotation ambiguity. Не переоцениваем gold. |
| 15014 | Добавлены варианты mortgage loss mitigation (`implicit_true`); общая тема совпадает, полнота поддержки не проверена. |
| 12855 | Придуманы температура, минуты на фунт и resting time для tenderloin; similarity всё ещё выше calibrated cutoff. |
| 14190 | Gold `implicit_true` для вывода о различиях Federalists; evidence «не были едины» делает границу поддержки неоднозначной. |
| 12200 | К почти скопированному длинному caution list добавлен неподдержанный фрагмент; support этого sentence≈0.934. |
| 13103 | Новое время приготовления lamb рядом с релевантными cooking terms; время preheat в evidence не подтверждает cooking duration. |

Числовые flags не ловят неверное присваивание уже имеющегося числа (14656), единицы, интервалы,
даты или арифметику. Negation presence/mismatch не определяет scope и логическое противоречие
(13118). Похожие слова могут дать высокий support unsupported claims (12200), а парафраз —
ложную тревогу (17081). Annotation ambiguity и implicit_true — реальные пределы интерпретации.
Примеры медицинской тематики здесь анализируются как записи датасета, не как рекомендации.

Автоматические, не причинные и не взаимоисключающие tags по всем ошибкам:
FP 281: number_present 208, negation_present 127, number_absent_from_retrieved 169,
oov_sentence 9; FN 111: соответственно 72, 45, 32, 0.
Наличие тега не доказывает причину ошибки; ручные категории выше не экстраполируются на все rows.

## Phase 2B: рекомендация

Следующий небольшой эксперимент — NLI на тех же top-3 retrieved chunks: premise=evidence chunk,
hypothesis=response sentence. Кандидат —
[cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small):
author model card указывает Apache-2.0 и обучение на SNLI/MultiNLI; это не RAGTruth-specific
detector. Не скачан и не запущен. До запуска закрепить revision/tokenizer/label mapping,
token budget, batching, truncation и измерить CPU/GPU memory/throughput.

Предлагаемый первый контроль: сохранить retrieval и segmentation Phase 2A, заменить только
cosine support на entailment probability, max по chunks и max risk по sentences.
Порог снова выбирается на calibration по прежнему exhaustive macro-F1/tie-break rule.
Так сравнение покажет эффект semantic scoring, хотя известные fragment/refusal ошибки останутся.
Отдельная следующая ablation — claim segmentation/refusal handling; определить правила на
train/calibration, не подбирать их по отдельным перечисленным dev примерам.

Модель NLI может помочь с парафразами и противоречиями, но это гипотеза: числовая точность,
неверная связь сущностей, retrieval misses и label ambiguity остаются рисками. Короткие chunks
могут не вместить поддержку из нескольких пассажей. Не обещаем прирост метрик или latency.
Dev уже просмотрен: будущие изменения — exploratory, final claims требуют отдельно разрешённого
untouched test. Решение о точной NLI модели/resource budget и claim-handling ablation ещё впереди.

## Изменённые/добавленные файлы Phase 2A

- Runtime: `src/grounded_hallucination_lab/baselines/lexical.py`, `eval/selection.py`,
  `eval/scored.py`, `lexical_experiment.py`, `cli.py`.
- Config/engineering: `configs/lexical.toml`, `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`.
- Tests: `tests/test_lexical.py`, `tests/test_scored.py`, `tests/test_lexical_cli.py`.
- Docs: `README.md`, `docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/EVALUATION_PROTOCOL.md`,
  `docs/PHASE2A_DESIGN.md`, `docs/PHASE2A_RESULTS.md`.
- Small text-free reports: `reports/lexical_calibration.json`, `reports/lexical_dev.json`,
  `reports/lexical_dev_reproduced.json`, `reports/phase2a_integrity.json`.
- Ignored local artifacts: `models/phase2a/{tfidf.joblib,frozen.json}`,
  `predictions/phase2a/{calibration.jsonl,dev.jsonl,errors.json}`, repeat evaluation predictions.

Исходные raw/processed datasets, модели, cache и full-text predictions не добавлены в Git.
Репозиторий ещё без initial commit; staged только workflow. Список выше — изменения этой фазы,
не результат сравнения с существующим Git commit.

## Точное воспроизведение

Из корня репозитория, PowerShell; существующий raw snapshot не меняется:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:GROUNDCHECK_RAGTRUTH_DIR = "data/raw/ragtruth"
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m grounded_hallucination_lab.cli baseline --data-dir data/raw/ragtruth --config configs/baseline.toml --output reports/majority_repro.json
.\.venv\Scripts\python.exe -m grounded_hallucination_lab.cli lexical-calibrate --data-dir data/raw/ragtruth --config configs/baseline.toml --lexical-config configs/lexical.toml --artifacts-dir models/phase2a-repro --predictions-dir predictions/phase2a-repro --output reports/lexical_calibration_repro.json
.\.venv\Scripts\python.exe -m grounded_hallucination_lab.cli lexical-evaluate --data-dir data/raw/ragtruth --config configs/baseline.toml --lexical-config configs/lexical.toml --artifacts-dir models/phase2a-repro --predictions-dir predictions/phase2a-repro --output reports/lexical_dev_repro.json
git ls-files --stage .github/workflows/ci.yml
```

Если phase2a-repro уже существует, используйте новое имя artifacts-dir для обеих команд.
Первичная загрузка при отсутствии snapshot: `python scripts/fetch_ragtruth.py` в активированном
venv; fetch требует сети, сами benchmark commands работают offline. Linux использует
`.venv/bin/python`. В новом checkout freeze создаётся заново: code/dependency hashes обязаны
совпадать между calibration и evaluation. Нельзя загружать чужие недоверенные joblib artifacts.
