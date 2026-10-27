# R7 — Graph-based evidence grounding (2026-10-24)

Пилот выполнен полностью, без обучения и без outer calibration/dev/test.
На новом synthetic suite автоматический граф улучшает macro-F1 относительно
R6-B на **+.4583**, относительно keyed retrieval на **+.2046**. Но natural
structural coverage = **0/648**. Количественный synthetic pilot gate пройден;
более строгий gate для рекомендации graph-aware training **не пройден**.
Мы сохраняем узкий структурный verifier как экспериментальный компонент,
не продвигаем его в основной detector и не объявляем исследование завершённым.

## Что проверено

Frozen R6-B42 SHA256
`86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e`.
FP32, batch1, TF32 off, response risk >=.37612172961235046; token cutoff .5.
Checkpoint, max-token aggregation и thresholds не изменялись. Plan/source/suite
freeze сохранён до predictions: EXPERIMENT_R7_PLAN.md, models/phase2b_r7/freeze.json.
Все восемь inference stages завершились с exit0: пять controls, три natural.

Natural: неизменные648 ответов/108 inner sources,442 supported/206 hallucinated.
Это многократно изученный development set. Controls:1536 ответов/16 новых
fictional parents,768 supported/768 contradicted,768 counterfactual pairs.
Шесть distractor variants × short/long × два порядка фактов × два значения
в ответе × два target-world assignments. Новых фактов из supervised fitting нет;
source disjointness проверена. Оба конкурирующих значения всегда присутствуют
в evidence, но truth определяется entity + property + qualifiers.

Для causal different/one_same/repeated неизменны target claim/fact/label,
число объектов, порядок, позиция и длина evidence в characters. Из256 групп
у8 остаётся разница BPE prefix length до2 tokens — измеренный residual confound.
Остальные варианты отдельно проверяют похожие имена, совпадающее свойство и
разнесённые passages. Длинные ответы содержат24 правильных filler sentences;
позиция target crossed beginning/middle/end. Это искусственная проза, не natural.
Шесть evidence styles, из них два специально вне грамматики extractor (4/16 parents).
Составитель grammar и suite один; это НЕ слепой внешний extraction benchmark.

Полные inputs прошли CPU preflight: max985 natural/378 controls tokens при
лимите2048; answer/evidence coverage1, ни одного truncation. Никакой gold span
или relation ledger не участвует в automatic retrieval/inference.

## Представление и варианты

`Entity --property [time, scope, condition, negation]--> Value(value, unit)`.
Каждое ребро имеет document ID, passage ID, clause [start,end) и value [start,end).
Равное число может быть общим value node, но subject edges не сливаются.
Alias links разрешаются только из явных identity statements; циклы/конфликты
дают ambiguous. Question не является evidence. Код: research/evidence_graph.py.

- **Baseline:** исходный R6-B на полном evidence/answer.
- **Keyed:** exact entity + property lookup, исходные fact clauses и alias clauses
  с provenance; дополнительный neural pass с тем же полным answer. Меняются
  только parsed claim token scores, остальной ответ сохраняет raw baseline.
- **Graph:** exact relation/value/qualifier matching. Подтверждённый scalar claim
  получает0 на своей clause; конфликт получает1 только на numeric phrase.
  Unknown/ambiguous и остальные токены сохраняют neural fallback. Эти0/1 —
  symbolic decision scores, НЕ calibrated probabilities.
- **Automatic:** extractor получает только text + document ID, четыре полные
  scalar grammars с открытыми именами/property strings, без suite/gold imports.
- **Oracle:** отдельный ledger, построенный при создании worlds. Только controls,
  диагностический upper-bound; runner запрещает oracle natural. Проверены все
  relations/offsets/labels программно и просмотрены типовые текстовые случаи;
  независимого второго annotator не было.

Missing relation не означает contradiction. Различаются extraction unknown,
ambiguous linking, missing retrieval, incompatible qualifiers и фактический
конфликт. Любое unparsed numeric evidence с цифрами блокирует symbolic override.
Это НЕ полная проверка неизвестных конфликтов: number words/implicit relations
и общая семантика за пределами грамматики не покрыты. Компонент пока не production-safe
для произвольной прозы. Negation/conflicting qualifiers проверены unit tests;
основной benchmark использует только однозначные scalar conflicts.

## Результаты

95% intervals:2000 paired source bootstrap draws,seed42;16 parents controls,
108 natural. Все варианты одного parent остаются вместе. CI условны на выбранный
checkpoint, suite и правила; не учитывают seed variance/selection bias и не
превращают шесть родственных grammars в независимые natural domains.
AP — Average Precision, не trapezoidal PR-AUC. CM rows=true/columns=pred [0,1].
Pair accuracy: оба counterfactual ответа правильно классифицированы; claim gap:
средний risk(contradicted)−risk(supported) внутри проверяемой clause.

### controls

| Method | Macro-F1 [95% CI] | Positive F1 | P / R | FPR | AUROC / AP | CM [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.4250 [0.3947, 0.4548] | 0.1690 | 0.8571 / 0.0938 | 0.0156 | 0.6995 / 0.6906 | [[756, 12], [696, 72]] |
| keyed_auto | 0.6788 [0.6095, 0.7412] | 0.7091 | 0.6528 / 0.7760 | 0.4128 | 0.8064 / 0.8766 | [[451, 317], [172, 596]] |
| graph_auto | 0.8834 [0.7596, 0.9720] | 0.8707 | 0.9917 / 0.7760 | 0.0065 | 0.9746 / 0.9776 | [[763, 5], [172, 596]] |
| keyed_oracle | 0.7333 [0.6641, 0.7969] | 0.8000 | 0.6667 / 1.0000 | 0.5000 | 0.9781 / 0.9805 | [[384, 384], [0, 768]] |
| graph_oracle | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 / 1.0000 | 0.0000 | 1.0000 / 1.0000 | [[768, 0], [0, 768]] |

| Method | Pair accuracy | Claim gap | Short TP / gold hit / n | Token / char / exact span F1 | ms/response | responses/s | Peak VRAM / process RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0833 | 0.0770 | 35 / 15 / 384 | 0.0348 / 0.0397 / 0.0000 | 35.47 | 28.20 | 0.620 / 1.446 |
| keyed_auto | 0.3633 | 0.3770 | 304 / 298 / 384 | 0.6305 / 0.6245 / 0.2293 | 63.05 | 15.86 | 0.620 / 1.446 |
| graph_auto | 0.7695 | 0.7666 | 304 / 298 / 384 | 0.8591 / 0.8613 / 0.8483 | 36.14 | 27.67 | 0.620 / 1.445 |
| keyed_oracle | 0.5000 | 0.4829 | 384 / 384 / 384 | 0.7568 / 0.7506 / 0.3151 | 70.63 | 14.16 | 0.620 / 1.446 |
| graph_oracle | 1.0000 | 1.0000 | 384 / 384 / 384 | 1.0000 / 1.0000 / 1.0000 | 35.76 | 27.96 | 0.620 / 1.446 |

| Difference | Macro-F1 delta [95% paired CI] |
|---|---:|
| keyed_auto − baseline | +0.2537 [0.1753, 0.3310] |
| graph_auto − baseline | +0.4583 [0.3342, 0.5604] |
| keyed_oracle − baseline | +0.3083 [0.2233, 0.3875] |
| graph_oracle − baseline | +0.5750 [0.5452, 0.6053] |
| graph_auto − keyed_auto | +0.2046 [0.1170, 0.2930] |

### natural

| Method | Macro-F1 [95% CI] | Positive F1 | P / R | FPR | AUROC / AP | CM [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.7842 [0.7481, 0.8192] | 0.7092 | 0.6912 / 0.7282 | 0.1516 | 0.8735 / 0.7834 | [[375, 67], [56, 150]] |
| keyed_auto | 0.7842 [0.7481, 0.8192] | 0.7092 | 0.6912 / 0.7282 | 0.1516 | 0.8735 / 0.7834 | [[375, 67], [56, 150]] |
| graph_auto | 0.7842 [0.7481, 0.8192] | 0.7092 | 0.6912 / 0.7282 | 0.1516 | 0.8735 / 0.7834 | [[375, 67], [56, 150]] |

| Method | Pair accuracy | Claim gap | Short TP / gold hit / n | Token / char / exact span F1 | ms/response | responses/s | Peak VRAM / process RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | n/a | n/a | 14 / 6 / 20 | 0.5800 / 0.5822 / 0.0546 | 40.47 | 24.71 | 0.644 / 1.442 |
| keyed_auto | n/a | n/a | 14 / 6 / 20 | 0.5800 / 0.5822 / 0.0546 | 41.25 | 24.24 | 0.644 / 1.443 |
| graph_auto | n/a | n/a | 14 / 6 / 20 | 0.5800 / 0.5822 / 0.0546 | 42.99 | 23.26 | 0.644 / 1.443 |

| Difference | Macro-F1 delta [95% paired CI] |
|---|---:|
| keyed_auto − baseline | +0.0000 [0.0000, 0.0000] |
| graph_auto − baseline | +0.0000 [0.0000, 0.0000] |
| graph_auto − keyed_auto | +0.0000 [0.0000, 0.0000] |

Shared-value gate slice (все кроме different):1280/16,640 positives.
Baseline/keyed_auto/graph_auto macro-F1 = .4121/.6769/.8814;
graph−baseline = +.46925,95%CI[+.34198,+.57009]. Pair accuracy
.69531; CM graph [[636,4],[146,494]], baseline [[631,9],[589,51]].
Все17 preregistered pilot checks выполнены, включая extraction P/R и retention.
Training gate дополнительно требует >=5% natural structural coverage:0% → fail.
Нет изменения критериев после результатов (reports/phase2b_r7/gate.json).

Localization — custom scorer, не официальный RAGTruth: micro token(any gold-char
overlap), non-whitespace character и exact trimmed/merged span при .5. Gold hit
означает хотя бы один положительный токен на gold span, а не просто response TP.
На controls short-error denominator384 (все long positives), natural20
(>=500 chars, gold nonspace fraction<=5%). Graph−keyed token F1 +.2286,
95%CI[+.1644,+.2914]; exact-span F1 +.6190,[+.4568,+.7426]. Но short-error TP
и gold-hit у graph/keyed одинаковы:304/384 и298/384; улучшение span F1 связано
с точными границами и меньшим числом FP, не новым recall поверх keyed.
Полные absolute/paired intervals всех метрик — analysis_controls/natural.json.

Natural predictions и все token probabilities **точно** воспроизвели R6-B:
max score difference0, ID order совпадает. Обе автоматические ветки тоже
token-for-token равны baseline. Natural642 sensitivity без15181: macro-F1 .783628
у всех. Не называем это переносом: extractor не применился ни разу.

Суммарное время восьми resident-model pipelines451.05s; измерения включают
extraction, tokenization, все neural passes и fusion, исключают model loading,
disk writes и analysis. GPU inference выполнялся последовательно, без training.
VRAM — PyTorch peak allocated; RSS — lifetime process peak, включая загрузку.
Небольшие timing differences не проверялись повторными performance trials.

## Extraction и причины ошибок

| Этап | Automatic controls | Oracle controls | Natural automatic |
|---|---|---|---|
| Evidence fact extraction | TP5760,FP0,FN1920; P1.00/R.75 | ledger7680/7680 по построению | 0 facts; gold evidence graph отсутствует |
| Claim extraction |1536/1536,FP0 |1536/1536 по построению |0/648 |
| Explicit aliases |1536/1536,FP0 |1536/1536 по построению |0 решений |
| Structural coverage |1152/1536=75% |1536/1536 |0/648 |
| Extraction unknown |384/1536=25% |0 |648/648 claim unknown |
| Ambiguous match/link |0 (проверено отдельно тестами) |0 |n/a |
| Keyed target hit |1152/1536;0 misses при извлечённом target |1536/1536 |648 fallback |
| Wrong resolved relation |0/1152 |0/1536 |n/a |

Все1536 controls имеют unparsed qualitative evidence clauses; это отдельно
от384 numeric extraction failures. У768 long answers есть unparsed filler.
Нельзя трактовать любую unknown clause как отсутствие всего evidence.
Natural extraction activations=[]: случаев для ручной проверки срабатываний нет;
natural extraction precision/recall неизвестны, а не0 или1.

Оставшиеся graph_auto ошибки —172FN и5FP, все на двух неподдерживаемых styles,
полностью унаследованы от neural fallback. На четырёх покрытых grammars graph
macro-F1=1.0; на family4/5 = .4812/.3753, как baseline. Это граница грамматики,
не доказательство общего reasoning. Oracle1.0 ожидаем на задачах exact ledger
comparison; это не достижимый автоматически natural результат.

Keyed_auto исправляет многие FN, но создаёт312 FP после корректного retrieval;
ещё172FN/5FP — fallback. Даже oracle retrieval даёт384FP при нулевых FN.
Следовательно, extraction/retrieval не единственный bottleneck: neural verifier
на укороченном evidence плохо подтверждает часть правильных claims при старом
пороге. Это может включать context-distribution shift и alias reasoning; текущая
абляция не разделяет эти механизмы. Порог по diagnostic не перенастраивали.

Graph исправляет531 baseline ошибок без новых; natural fixed/new =0/0.
Конкретные примеры (полные тексты/offsets/provenance в error_stage_audit.json):

- `r7:Lunora:different:l0:o0:a0:w1`: target HubA=719, Site — его alias,
  ответ713; у чужого HubB действительно713. Baseline .06138 ошибочно принимает,
  keyed .93252 и graph1 обнаруживают конфликт. Numeric phrase `713 crates`
  локализована по оригинальным offsets.
- `r7:Lunora:different:l0:o0:a0:w0`: тот же supported answer713 при HubA713.
  Baseline .00918 верен, keyed .45944 создаёт FP, graph0 сохраняет support.
  Long аналог: raw .00760, keyed .74547, graph .00760.
- `r7:Parnel:different:l0:o0:a0:w1`: evidence `HubA carried a evening stock
  tally of 759 crates`, ответ753. Auto не знает эту grammar и сохраняет FN
  .21133; oracle-keyed .93528, oracle-graph1. Это extraction failure, не
  отсутствие evidence и не найденное логическое противоречие automatic graph.
- Supported long Parnel `...:l1:o0:a0:w0`: raw/graph_auto .48984 — FP;
  oracle graph .00228 исправляет. Удалять этот случай или дорабатывать parser
  по нему после freeze не стали.

## Causal distractor audit

В каждой строке128 contradicted ответов/16 parents; точный target/answer/label
не меняются. Mean claim-local delta совпадает с response delta на этой positive
slice. Интервалы paired source95%, все draws те же seed42.

| Нерелевантное изменение | Метод | Mean risk delta [95%CI] | TP→FN / FN→TP |
|---|---|---:|---:|
| Один distractor получает answer value | baseline |−.04081 [−.05487,−.02735] |10 /1 |
| То же | keyed_auto / graph_auto |−.00953 [−.02079,−.00024] |2 /0 |
| Два distractors получают answer value | baseline |−.03896 [−.06013,−.01849] |12 /3 |
| То же | keyed_auto / graph_auto |−.01258 [−.03427,−.00044] |4 /1 |

На supported128 baseline risk тоже снижается: one_same−.01725,
repeated−.02178; это не специфическое распознавание противоречий. При repeated
исправлены3 FP и добавлен1 FP. Auto graph:1 исправленный/1 новый FP в этом
label-preserving contrast, оба в fallback; это отличается от сравнения graph
с baseline на одинаковом input, где новых ошибок нет.

При извлечённом target keyed и graph удаляют влияние distractors на claim.
Оставшаяся causal чувствительность приходит из unsupported-style fallback;
оба oracle режима сохраняют claim score при изменении distractors. Поэтому
graph выигрывает у keyed главным образом меньшим supported FPR и точными
span boundaries, а не большим short-error recall или уникальной distractor
инвариантностью. Не выдаём deterministic relational join за пользу GNN.

Variant macro-F1 baseline/keyed_auto/graph_auto:
different .4854/.6879/.8935; one_same .4250/.6805/.8854;
repeated .4274/.6768/.8814; similar_names .4003/.6731/.8773;
same_property .3834/.6736/.8771; split_passages .4227/.6805/.8854.
Все category breakdowns, включая long/order/claim position и natural generator,
сохранены в analysis JSON. Разница с R6 transfer .8638 не является ухудшением
весов: это иной, более сложный suite с explicit aliases и иной прозой.

## Решение и следующая проверка

**Graph-aware training сейчас не рекомендуем.** Узкий scalar binding bottleneck
устранён при корректно извлечённых фактах, но natural applicability не получена.
Простого keyed retrieval тоже недостаточно: он повышает FPR до.4128 в automatic
и.50 в oracle режиме. Оставляем оба как исследовательские ablations; отказываемся
от продвижения графовой гипотезы в основную модель на основании этого пилота.

Следующий наиболее информативный пакет — заранее размеченный небольшой
train-side набор естественных numeric relations и aliases с evidence spans,
отдельными extraction-fit/evaluation parent groups и held-out renderers.
Сначала измерить relation extraction P/R, qualifier/link errors и coverage
независимо от hallucination head; сравнить sentence retrieval с keyed retrieval,
сохраняющим соседний контекст. Не превращать R7 ошибочные IDs в parser exceptions.
Порог успеха и состав freeze до inference; outer не открывать.

Условная минимальная graph-aware архитектура, если coverage gate впоследствии
пройден: существующий encoder, pooling по извлечённым entity/property/value
mentions и маленький relation-support MLP рядом с неизменным token head.
Контроль — тот же encoder+MLP без связей на тех же данных; supervised relation CE
по проверенной разметке и прежний token CE, без GNN/новых ranking losses.
Она была бы нужна для переноса binding между формулировками, а не для замены
тривиального exact-value comparison. Текущее R7 не даёт ни natural supervision,
ни coverage, оправдывающих такой fit; обучение не запущено.

Ограничения:16 parents и6 родственных grammars, искусственный alias syntax,
повторяющийся filler и только scalar conflicts; одинаковый автор parser/suite;
неполная обработка неизвестных number-word conflicts; reused natural inner;
hard0/1 overrides меняют score semantics, поэтому claim-gap рост не является
измерением calibrated confidence. Human-independent graph annotation отсутствует.
Оригинальность/SOTA/перенос на unseen real domains не заявляются.

## Проверки, сохранение и воспроизведение

- Offline pytest:197 passed/7 skipped,55.76s. Research opt-in:203 passed/1 skipped,
  73.14s. Единственный skip — legacy full-RAGTruth majority/dev regression.
- 14 новых graph tests: open-world fallback, qualifiers/negation/conflicts,
  alias/provenance, automatic/oracle separation, все labels/offsets и core matching.
- Ruff passed; pip check обеих venv: No broken requirements found.
- CI .github/workflows/ci.yml существует и tracked, Python3.11/3.12, offline pytest,
  Ruff/pip check. Remote workflow run не проверялся и не объявляется успешным.
- Старые111 R6 artifacts и исходный R6 snapshot сохранены и проверены по SHA256.
  Новый source ZIP и member hashes: research_snapshots/R7_source_20261024.zip/.json.
  Inventory: models/phase2b_r7/R7_artifact_manifest.json. Незавершённых fits нет.

Основные изменения: src/grounded_hallucination_lab/research/evidence_graph.py,
graph_suite.py; scripts/research_r7.py; tests/test_evidence_graph.py;
EXPERIMENT_R7_PLAN.md; docs/PHASE2B_R7_RESULTS.md, EVALUATION_PROTOCOL.md,
PROGRESS.md, DECISIONS.md; RESEARCH_NOTES.md. Analysis/queue/seal helpers находятся
в models/phase2b_r7/*.py и также внутри НЕ ignored source ZIP, поэтому не теряются
из-за .gitignore. Dataset texts, predictions, веса и caches остаются ignored;
ничего не committed/pushed/published.

Проверки в текущей рабочей копии (не запускают новый benchmark):

```powershell
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m pytest -q
$env:GROUNDCHECK_RESEARCH_MODELS='1'
.venv-research\Scripts\python.exe -m pytest -q
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pip check
.venv-research\Scripts\python.exe -m pip check
```

Полное повторение — только в **отдельной восстановленной рабочей копии** frozen
source ZIP, с прежними локальными R6 checkpoint/config/manifest, inner records и
raw file hashes. ZIP содержит код/документы, не weights/dataset. При восстановлении
исключить reports/ и прежние R7 outputs; helpers models/phase2b_r7/*.py сохранить.
Установить frozen research environment (configs/research-lock.txt); PYTHONPATH
должен указывать src именно восстановленной копии. Старые outputs текущего repo
не удалять: write_json/write_rows намеренно откажутся их перезаписывать.

```powershell
$env:PYTHONPATH="$PWD\src"
$py='.venv-research\Scripts\python.exe'
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
& $py scripts/research_r7.py freeze
& $py models/phase2b_r7/preflight.py
& $py models/phase2b_r7/run_packet.py
& $py models/phase2b_r7/analyze.py
& $py models/phase2b_r7/structural_analysis.py
& $py models/phase2b_r7/error_audit.py
& $py models/phase2b_r7/gate_tables.py
```

Source/dependency mismatch должен остановить запуск. Не обходить старые freezes.
Все numerical reports содержат точные denominators и конфигурацию; никакого
нового training seed, dev leaderboard, final test или release в R7 нет.
