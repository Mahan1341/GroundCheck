# R5 — Learning Numerical Equivalence

2026-10-24. Matched seed42 выполнен. **Raw+canonical CE не прошёл заранее
зафиксированный transfer/retention gate. Seeds43/44 не запускались.** Это конец
данного экспериментального пакета, исследование проекта продолжается.

## Главный результат

Все значения ниже — response-level macro-F1, include/all, одинаковые IDs внутри
каждого столбца, fixed threshold **0.37612172961235046**, positive iff score>=threshold.

| Метод | Natural648 | R3 diagnostic640 | R4 diagnostic864 | Новые R5 controls672 |
|---|---:|---:|---:|---:|
| Frozen B7 | .7955 | .5107 | .5090 | .4796 |
| Frozen R3-A | .7799 | .6712 | .6597 | .5222 |
| R4 symmetric inference, R3-A weights | .7799 | .8292 | .7837 | .5267 |
| **R5-A: raw+raw CE** | **.7912** | .8115 | .6608 | **.5478** |
| **R5-B: raw+canonical CE** | .7814 | **.8773** | .6956 | .5191 |

На новых контролях B−A = **−.0288**, paired source95% CI **[−.0545, −.0006]**.
На natural B−A = −.0098, CI [−.0291,+.0092]: статистически уверенного общего
падения macro-F1 нет, но noninferiority gate не выполнен. Natural FPR вырос
на .0271, CI [.0112,.0454]. B отстаёт от B7 на .0140 по natural point estimate.

Улучшение на старых synthetic suites реально в рамках этих наборов:
R3 B−A +.0657 [.0385,.0902], R4 +.0348 [.0246,.0463]. Для R3 консервативная
кластеризация по четырём повторяемым именам сохраняет положительный CI
[.0409,.0800]. Это **не устойчивое освоение числовой эквивалентности**:
перенос на новую прозу/форматы и сохранение natural quality одновременно не получены.

Полные таблицы всех пяти методов, positive F1, supported FPR, positive/contradiction
recall, pair accuracy, separation, token/span F1 и интервалы:
`measured_tables.md` (historical local artifact; not distributed).
Все accuracy, precision/recall/F1, AUROC, AP, CM, denominators, category slices,
абсолютные интервалы и парные разности сохранены в
`reports/phase2b_r5/{method}_{dataset}.json` и `analysis_{dataset}.json`.
AP здесь average precision, не трапецеидальная PR-AUC.

## Что именно обучалось и проверено

Обе модели стартовали с одного B7 inner42 checkpoint, hash
`7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4`.
2568 natural inner-fit responses/428 sources плюс960 R3 authored fit/18 sources.
3528 уникальных записей,446 групп; по два views на запись. Один epoch,
3528 microbatches,882 optimizer updates, seed42. В B второй view canonical,
в A второй view raw. Никакого consistency/ranking loss, изменения encoder,
retrieval, segmentation, max aggregation или порога.

AdamW lr1e-5, weight decay.01, warmup10%, clip1, batch2/accumulation4,
CE weights[1,3.370202392225359], BF16 autocast/FP32 weights, optimizer и loss,
gradient checkpointing. Папка `best` содержит последний epoch, выбора по метрикам
не было. Только A/B — matched ablation: frozen R3-A имел другую training exposure.

Проверка конфигов обнаружила только ожидаемые различия `arm` и `targets_sha256`.
Порядок записей совпал, SHA256
`1231952201ed23dd48589a471870b2745d977321cf1ac229e3e6f738764bab7c`.
Повторяемые raw views не являются независимыми наблюдениями.

Использован прежний conservative R4 parser. Он преобразует точные целые
числа/поддерживаемые units, оставляет неоднозначные/range/decimal случаи.
Original gold intervals отображаются в canonical character intervals;
после retokenization targets пересчитываются. Все annotation texts, labels,
answer-token targets и loss masks проверены на полном training input.
Prefix/question/evidence/special tokens исключены из answer loss.
Gold spans не выбирают inference evidence или окна.

- Ответ изменён у149 записей, evidence у252; хотя бы одно поле у400/3528.
- В natural train изменены лишь16/2568 записей:5 answers,12 evidence.
- Positive answer tokens34476→34380; все answer tokens426889→426697.
- Расширений annotation intervals0. Single-window membership совпадает у всех
  записей, max input1008 tokens, answer/evidence coverage100%, truncation0.
- Следовательно, утверждать одинаковые token-target массивы нельзя. Совпадают
  исходные фактические labels и supervision policy, а не длина токенизации.

CE на canonical views также меняет частоту поверхностных форм: на затронутых
записях raw вид встречается один раз вместо двух. Это часть проверяемого рецепта,
а не изолированный механизм принудительной инвариантности.

## Контрольные наборы и статистика

План `EXPERIMENT_R5_PLAN.md` (historical local artifact; not distributed), source snapshot, datasets
и checkpoints frozen **до profiling/training**. Plan SHA256
`2d36f213f43f80c2f9e5ad5a44faef57d7d9b0898454c5da3e934523b8272c51`.

| Набор | Responses | Positive / negative | Source clusters | Назначение |
|---|---:|---:|---:|---|
| Natural inner-val | 648 | 206 / 442 | 108 | Reused natural development quality |
| R3 diagnostic | 640 | 320 / 320 | 12 nominal, 4 names | Исторические controls |
| R4 diagnostic | 864 | 432 / 432 | 12 | Исторические controls |
| Новые R5 | 672 | 336 / 336 | 12 | Новые fictional parents, formats, prose |

R5: по96 примеров/48 пар на каждую из семи категорий. Новые receipt/interview
families, spaced word numbers, exact integer-unit conversions, decimal exact
representations, paraphrases, first/second property distractors и короткие
ошибки среди36 правильных filler sentences. Same-answer pairs имеют разную
истинность при контролируемой перестановке evidence relations. Число остаётся
в другом entity/property: его наличие в документе не доказывает поддержку.
Независимый Decimal/word parser проверил величины и labels всех672 examples.
Decimal forms намеренно вне области canonicalizer. Категория exact_units
содержит quantity identity controls; реальные unit conversions — duration/distance.

Все parents исключены из supervised fit. «Новые formats» означает отсутствие
в authored augmentation recipes; не доказано отсутствие в pretraining/natural train.
Повторные варианты source не считаются независимыми: 2000 paired source bootstrap
resamples, seed42, одинаковые draws для всех систем. Интервалы conditional на
фиксированных weights/thresholds; не учитывают training-seed variation, выбор
метода на reused inner data или множественные exploratory comparisons.

Outer calibration/dev/test не оценивались и не использовались для выбора.
Natural642 sensitivity исключает только известный source15181 с shared passage:
MF1 A=.790690, B=.780855, B7=.795103; вывод сохраняется. Это secondary sensitivity,
не изменение primary membership. Natural B7 .7955 относится к fixed batch2;
исторические .7959 были batch1. Порог не менялся, сравнение всех R5 arms — batch2.

## Перенос, ошибки и localization

| Новая категория | A macro-F1 | B macro-F1 | A/B contradiction recall | A/B supported FPR |
|---|---:|---:|---|---|
| digits | .4454 | .3992 | .1250 / .0625 | .0417 / .0000 |
| spaced_words | .3333 | .3333 | .0000 / .0000 | .0000 / .0000 |
| decimal_units | .5218 | .5157 | .8542 / .9167 | .7292 / .7708 |
| exact_units | .5071 | .5804 | .6875 / .6667 | .6458 / .5000 |
| paraphrase | .5953 | .4958 | .3542 / .1667 | .1042 / .0000 |
| property | .4812 | .4565 | .1667 / .1458 | .0417 / .0625 |
| long_short | .6042 | .4332 | .6042 / .1042 | .3958 / .0208 |

**Ranking и operating point расходятся.** На R5 new AUROC A=.5787→B=.6127,
AP .5460→.5664; интервалы разностей [.0216,.0443] и [.0086,.0303]. Доля пар
с higher risk у противоречия .7083→.7798, CI разности [.0387,.1071]. Но
both-correct pairs49/336→37/336, mean score separation .0696→.0857
(CI разности [−.0029,.0310]). При fixed threshold FN202→237, FP94→65;
CM A=[[242,94],[202,134]], B=[[271,65],[237,99]]. Lower FPR достигнут ценой recall.
Это основание диагностировать перенос шкалы риска вместе со слабым relation
grounding, но **не разрешение подобрать rescue threshold на этой диагностике**.

Natural: CM A=[[392,50],[65,141]], B=[[380,62],[61,145]], positive F1
.7103→.7022, AUROC .8745→.8708. Исправлено9 решений, добавлено17 ошибок.
Новые controls:41 исправление,47 новых ошибок. R3:51/12; R4:40/10.

**Short error в длинном ответе не решён.** R5 long_short response TP29/48→5/48,
paired recall-delta CI [−.7500,−.2495]; попадания token probability>=.5 в gold
span всего1/48→4/48. Уменьшение spurious max не равно надёжной детекции ошибки.
На natural predeclared slice20 positives: TP12/20 и gold hits6/20 у обеих моделей
(B7:15/20 и9/20). Здесь >=500 characters и доля gold **nonwhitespace** characters<=5%;
это не прежний R4 slice23 с другим знаменателем.

Micro token F1 natural .5733→.5788, exact-span .0418→.0465; оба paired CI
разностей включают0. На R5 new token F1 .3531→.3595, exact-span .2281→.2068;
CI разностей [−.0214,.0408] и [−.0503,.0090]. На R4 synthetic localization
улучшилась заметно, но short-error gold hits39/144→27/144: aggregate F1 не
заменяет разбор кратких ошибок. Scorer custom, не official RAGTruth:
token any gold-character overlap; character nonwhitespace; exact-span trimmed
overlap/whitespace-adjacent merged intervals; fixed token threshold.5. R4sym
projected на тот же original token grid, с возможным расширением numeral ranges.

Примеры (полные texts, gold, scores и evidence сохранены в ignored
`predictions/phase2b_r5/errors_B_minus_A_{dataset}.jsonl`):

- `r5:Willowmarsh:exact_units:o0:a1:e1`: evidence3km, правильный answer3000m;
  risk .4391→.1611, FP исправлен. Это конкретный выигрыш exact-unit equivalence.
- `r5:Pinehaven:decimal_units:o0:a0:e1`: target259 bolts, answer245.00;
  risk .1118→.4083, пропущенное противоречие найдено, хотя token threshold.5
  ещё не достигнут. Decimal category в целом всё равно имеет высокий FPR.
- `r5:Oakmere:paraphrase:o0:a0:e1`: target243, чужому Annex принадлежат229;
  answer229, risk .5870→.3677 — новый FN при точной entity-value ошибке.
- `r5:Quillford:property:o1:a1:e1`: правильный first shipment275, second261;
  risk .3099→.5666 — новый FP из-за property distractor.
- `r5:Oakmere:long_short:o1:a0:e1`: та же ошибочная229 внутри длинного ответа,
  gold interval[678,687), risk .3858→.2500. В обеих моделях max действительно
  находится на числе, но B опускает его ниже decision boundary.

У frozen R4 parser обнаружена конкретная template sensitivity: conservative
guard на `about` оставляет неизменным evidence вида `When asked about ...`.
Во всех48 interview/spaced_words examples answer нормализуется, evidence нет;
в48 receipt counterparts меняются оба. Изменять parser после результатов не стали.
Это объясняет ограниченность «symmetric» policy на новой прозе; даже исходные
digits/property cases с низким recall показывают более широкий relation bottleneck.

## Gate, ресурсы и сохранность

`reports/phase2b_r5/seed42_gate.json`: **false**. Не пройдены new macro gain/lower CI,
pair-accuracy gain, natural noninferiority lower bound и retention относительно B7.
Пройдены остальные зарегистрированные условия, включая natural short-error
retention и ограничения localization loss. Seeds43/44 и rescue tuning не запускались.
Отрицательный вывод относится к этому CE-view recipe, не ко всем возможным
способам обучения numerical grounding.

| Run | Training seconds | Updates | Peak allocated / reserved GiB |
|---|---:|---:|---|
| R5-A42 | 650.60 | 882 | 2.956 / 3.135 |
| R5-B42 | 641.57 | 882 | 2.956 / 3.135 |

Оба profiling runs и отдельный longest genuinely changed-pair probe прошли finite
loss/gradients/<11GiB gate. Два обучения суммарно21.54min, без OOM и truncation.
На new672 inference A/B14.17/14.16s,47.41/47.47 responses/s; natural648
14.46/14.91s,44.82/43.47 responses/s. Peak inference allocated .901/.953GB.
Это batched model-pipeline timings, **не** end-to-end single-request latency:
загрузка weights, tokenization, projection, hashing и disk I/O не включены.
Runtimes и p50/p95 batch forward latency сохранены по каждому методу/набору.
Один GPU-heavy процесс за раз, локальные checkpoints, никаких платных API/downloads.

Checkpoints:
- `models/phase2b_r5/R5As42/best/model.safetensors` —
  `085c2941e7afac27cafd387df914748b851d04d92afa3c70820d8f29ab723de3`.
- `models/phase2b_r5/R5Bs42/best/model.safetensors` —
  `5f75e548d86a5a33935f3c06e70156354bc87861c0b74e1926a6ad9541bc67e8`.

Freeze contains train/data/model/source/dependency hashes. Старые235 R4 artifacts
проверены; old reports/predictions/checkpoints не перезаписаны. Reused baseline
predictions имеют verified hashes и одинаковые IDs; старые source guards не обходили.
Итоговый inventory `models/phase2b_r5/R5_artifact_manifest.json`, восстановимый
source/methodology archive `research_snapshots/R5_source_20261024.zip` с companion
JSON. Weights/dataset texts/predictions не попадают в source archive и git.

## Проверки и воспроизведение

186 passed в research env, включая opt-in tokenizer/models/data checks (69.30s).
Offline179 passed/7 skipped (58.82s). Ruff passed, `pip check` passed в обеих venv.
CI `.github/workflows/ci.yml` существует и tracked, matrix3.11/3.12; новые offline
tests входят через обычный pytest. Успешный remote workflow run не заявляется.
Отдельно проверен отказ repeat gate: команда seed43 завершилась ожидаемым
`repeat-seed gate closed` до загрузки модели/создания checkpoint directory;
новое обучение не начиналось (`repeat_guard_check.json`).

Выполненные команды из корня проекта (на существующих completed outputs повторно
не запускать: overwrite guards намеренно откажут):

```powershell
.venv-research\Scripts\python.exe scripts/research_r5.py freeze
.venv-research\Scripts\python.exe scripts/research_r5.py profile --arm A
.venv-research\Scripts\python.exe scripts/research_r5.py profile --arm B
.venv-research\Scripts\python.exe models/phase2b_r5/profile_changed.py
.venv-research\Scripts\python.exe models/phase2b_r5/audit_controls.py
.venv-research\Scripts\python.exe scripts/research_r5.py train --arm A --seed 42
.venv-research\Scripts\python.exe scripts/research_r5.py train --arm B --seed 42
.venv-research\Scripts\python.exe models/phase2b_r5/evaluate_all.py
.venv-research\Scripts\python.exe models/phase2b_r5/make_tables.py
.venv-research\Scripts\python.exe models/phase2b_r5/check_repeat_guard.py
$env:GROUNDCHECK_RAGTRUTH_DIR="data/raw/ragtruth"
$env:GROUNDCHECK_RESEARCH_MODELS="1"
.venv-research\Scripts\python.exe -m pytest -q -o cache_dir=models/r5_verify_cache --basetemp=models/r5_verify_tmp
.venv-research\Scripts\ruff.exe check .
.venv-research\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pip check
.venv-research\Scripts\python.exe models/phase2b_r5/seal.py
```

Для нового полного replay: отдельный checkout из source archive, та же pinned
research environment, оригинальные raw files и локальные baseline checkpoints/
sealed R4 artifacts; **новый пустой R5 output tree**, без удаления текущих outputs.
`evaluate_all.py` вызывает пять `research_r5.py evaluate --method ...` последовательно,
потом `analyze.py`. Helpers из ignored models folder также включены в source archive.
Mid-epoch optimizer resume не реализован; оба обучения завершены, возобновлять нечего.
Не подменять source hashes старого freeze для запуска изменённого кода.

Изменены/добавлены: EXPERIMENT_R5_PLAN.md; scripts/research_r5.py;
research/numerical_views.py и numerical_views_suite.py; tests/test_numerical_views.py;
этот отчёт, EVALUATION_PROTOCOL.md, PROGRESS.md, DECISIONS.md, RESEARCH_NOTES.md.
Остальные новые файлы — локальные artifacts/helpers/snapshot. Ничего не опубликовано.

## Следующая проверяемая гипотеза

Главное ограничение теперь конкретнее: на новой prose модель плохо проверяет
**entity–property–value relation даже с обычными цифрами**; canonical views
улучшают часть ranking, но не дают стабильных detection/localization. Одними
форматами чисел это не исправлено.

Один следующий matched experiment: **разнообразие relation templates при неизменных
числовых формах**. Из B7 сравнить CE с повтором прежних authored templates против
CE с тем же количеством записей, теми же train parents/values/labels, но разными
extractive relation renderers и контролируемыми entity/property swaps. Natural
replay, optimizer, steps, seed и max aggregation одинаковы; никакой canonicalization
или consistency loss. Это известная supervised augmentation, не заявление новизны.
Новые parents и template families для проверки фиксировать до запуска; R5 new
после этого отчёта уже development history, не blind evaluation.

Критерий: рост both-correct relation pairs и short-error localization на новых
families одновременно с natural retention, с отдельным контролем score scale;
не спасать результат порогом на диагностике. Предварительный бюджет одного
matched seed — до30min на RTX3060 по измеренным R5 временам, с новым profiling.
Это предложение следующего пакета, обучение в данном пакете не запускалось.
