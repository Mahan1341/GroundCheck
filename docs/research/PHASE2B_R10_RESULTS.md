# R10 — Robustness without evidence blindness

2026-10-25. Ограниченный matched packet завершён; исследование продолжается.
**Permutation augmentation не прошла preregistered gate. Дополнительные seeds
не запускались. B7 остаётся исходным reference; R10-B его не заменяет.**

## Что проверили

`R10_RESEARCH_PROPOSAL.md` (historical local artifact; not distributed) и
`models/phase2b_r10/freeze.json` зафиксированы до fit. Один matched seed42:
A=CE raw+raw; B=CE raw+permuted, обе модели от одного B7 inner checkpoint.
2568 natural training responses/428 sources,789 positive;5136 exposures,
642 AdamW updates, одинаковые targets, порядок, class weights и optimizer.
Encoder, retrieval, segmentation, max-token aggregation и thresholds не менялись.
Новых synthetic templates, graph/parser changes, ranking/consistency losses нет.

До fit проверили смысл перестановок. Нельзя переставлять произвольные блоки с
`the first passage`, `above`, `then`, анафорическими продолжениями и неполными
recipe/procedure steps. После консервативного screen разрешены124/428 training
sources (744 changed second views) и34/107 inner sources (204 responses).
Остальные training examples остаются raw+raw в обоих arms. Нумерованные passage
headers перемещаются вместе с блоками; внутренний порядок каждого блока сохранён.
12 final eligible training sources просмотрены ассистентом; это не independent
human gold и не доказательство semantic equivalence всех разрешённых источников.
False exclusions и скрытые discourse dependencies возможны. Никаких ID-specific rules.

Все2568 raw/rotated input lengths совпали, max1008 tokens. Answer offsets/targets
побитово совпали; evidence/answer coverage1.0, prefix/special tokens исключены из loss.
`changed_second_views=744` в configs описывает потенциально изменённые views до
arm selection: фактически A использует0 rotated views, B744. Configs различаются
только полем arm. BF16 training; FP32/batch1/TF32off inference.

Natural evaluation:642 responses/107 reused inner sources,438 negative/204 positive.
Исторические partitions не изменены; известный partial-passage duplicate15181
исключён только из прежнего R9 sensitivity subset. Outer calibration/dev/test
не открывались. Response threshold >=0.37612172961235046, token >=0.5 неизменны.
Обе cyclic rotations оцениваются, лучшая не выбирается. Все сравнения парные.

## Главный результат и механизм

| Метод | Natural macro-F1 | Hallucination F1 | Token F1 | Order flips /204 | Short detected /19 | Short gold-hit /19 |
| --- | --- | --- | --- | --- | --- | --- |
| B7 frozen | 0.7956 | 0.7306 | 0.5870 | 18 | 14 | 8 |
| R10-A: extra CE control | 0.7875 | 0.7094 | 0.5769 | 13 | 11 | 6 |
| R10-B: permutation CE | 0.7844 | 0.7059 | 0.5796 | 10 | 11 | 6 |

1. **Decision-flip count недостаточен для вывода об invariance.** У B снижение
   относительно A составляет23.08%, меньше требуемых25%; парная разность flip-rate
   −1.47pp,95% CI[−3.43,+0.49]pp. Mean absolute response-score change почти одинаков:
   A0.0447,B0.0440; у B7 он даже ниже,0.0420. Нельзя приписывать18→10 только augmentation:
   extra CE control уже даёт18→13. Это совместимо со сдвигом score margins.
2. **Небольшая token-level invariance действительно наблюдается.** Mean absolute
   token-probability change A0.0154→B0.0144; paired delta95% CI
   [−0.0015484,−0.0005248]. Но этот эффект не превратился в убедительную устойчивость
   response max-score или лучшее factual grounding. Это условный результат одного
   seed на34 reused source groups, не независимое подтверждение.
3. **Natural retention не пройдена.** B−A macro-F1 delta95% CI[−0.0143,+0.0077];
   B−B7[−0.0356,+0.0121]. Они не доказывают значимый macro-F1 проигрыш, но и не
   подтверждают заранее требуемую non-inferiority с margin0.01. Recall обоих CE arms
   144/204 против160/204 B7; B−B7 recall delta95% CI[−0.1263,−0.0326].
   Оба arms теряют одинаковые short gold-hit IDs15663/15741. Основная потеря recall
   присутствует уже в control; приписывать её именно augmentation нельзя.
4. **Broad evidence use сохраняется, fine-grained binding слаб.** При удалении
   evidence средний risk shift на66 изначально negative ответах: B7+0.7071,
   A+0.7450,B+0.7622. Никаких новых gold labels/F1 для blank нет. При контролируемых
   factual swaps mean claim-local gap у B только0.0037 на R6 и0.0042 на R1;
   требуемый floor0.01 не достигнут. На R6 interval включает0. Модель реагирует на
   отсутствие evidence, но это не означает надёжного различения двух factual worlds.
5. **Не доказан отдельный augmentation-induced collapse.** B−A claim-gap intervals
   включают0 на обеих suites. Однако B−B7 gaps ниже: R6 delta95%
   [−0.0072735,−0.0000294], R1[−0.0110540,−0.0019984]. Control тоже ослабляет сигнал.
   Даже B7 имеет R6 gap0.0074 и pair accuracy7/384: это исходный bottleneck,
   а не проблема, впервые созданная R10. Нельзя назвать A сильным factual verifier.

Joint gate failed: order improvement, natural non-inferiority/retention, short-hit
retention против B7 и absolute factual sensitivity не пройдены. R1 ranking
non-inferiority тоже не подтверждена. Полная таблица каждого gate ниже.
Seeds43/44 не запущены; пороги, rotations и screened membership после результатов
не изменялись. Никакого threshold rescue.

## Ошибки и локализация

На natural B исправляет5 и создаёт7 response errors относительно A **по dataset
labels**. Post-hoc cases выбраны по наибольшему |B−A score| в обе стороны, плюс
оба lost short-hit. Они не входят в inference features или исключения.

- **14253, ложное ощущение исправления.** A risk0.3027→B0.4974 восстанавливает
  positive response decision. Но максимум B находится на `*` [259,260), вне gold;
  max probability на gold spans всего0.1670. Корректный response label здесь не
  подтверждает правильную локализацию/проверку фактов.
- **13626, новый FP.** Dataset-negative ответ объединяет recipe passages.
  A0.2321→B0.5624; peak на `45` [506,509), хотя этот предел времени присутствует
  в evidence. Источник procedural и не входит в order subset: эффект training
  распространяется также на неаугментированные типы примеров.
- **15663, короткое unsupported дополнение в длинном ответе.** Gold включает
  phone lock/unlock status. B7 max-gold0.5723→A0.3271/B0.3250. Peak всё ещё на
  gold `locked` [1142,1149), но confidence опускается ниже обоих frozen thresholds.
  Это потеря detection и localization при сохранённом приблизительном месте ошибки.
- **15741, response TP без локализации.** B7 max-gold0.6689→B0.3986; у B общий
  peak0.4594 на `that` [205,210), вне gold [151,192). Response остаётся positive,
  но token threshold0.5 не достигается. Нельзя оценивать localization по response F1.

Дополнительная annotation caveat: у16083 gold0, но ответ меняет условие
`with/without STI` относительно предоставленного passage. Поэтому его FP→TN
считается «исправлением» только по сохранённой разметке; assistant review не
подтверждает содержательную корректность. Labels не менялись. Это отдельный
кандидат на независимый review, не оценка частоты ошибок RAGTruth и не медицинская
рекомендация. Полные тексты доступны только в локальном ignored review pack.

R6 long-short: B claim ranking33/64=0.5156, mean gap≈0.0001, почти нет meaningful
separation. R1 negation: claim gap B7≈0.0080,A≈0.0080,B≈0.0019. Эти category slices
описательные, источников мало; без отдельных significance/causality claims.
На обеих controlled suites у B token F1=0 при fixed0.5; response positives
не являются подтверждением локализации фактического error slot.

## Статистика и ограничения

2000 paired source bootstrap resamples, seed42, percentile95. Natural groups107,
order groups34; R6 groups16/384 pairs, R1 groups15/75 pairs. Варианты одного parent
не считаются независимыми. Одни draws для моделей; абсолютные и paired intervals
response, localization и sensitivity доступны в `reports/phase2b_r10/analysis.json`.
AP — Average Precision, не trapezoidal PR-AUC. CM rows=true,cols=pred, labels[0,1].
Token scorer: any overlap с gold characters; character scorer: non-whitespace;
exact spans: trimmed/merged adjacent predicted intervals. Это custom micro scorers,
не официальный RAGTruth scorer. Short natural:>=500 chars и gold<=5% nonspace chars,
19 positive responses/16 sources. Потеря gold-hit проверяется по ID, а не только count.

Natural inner и authored suites многократно использовались ранее. Нет независимого
подтверждения, между-seed variance и поправки на историю выбора. Bootstrap этого
не исправляет. Строгий gate может быть underpowered на34 sources; это основание
не расширять эксперимент автоматически, а не доказательство бесполезности любой
augmentation. Отдельно не исследованы все перестановки, другие дозы CE или learning
rates; выбирать их по этим результатам как «прошедший R10» нельзя.

## Независимая рекомендация

**Не продолжать чистую order augmentation как основную research ветку.** Небольшое
сглаживание token scores есть, но исходная цель — grounded discrimination — не
достигнута. Я также не считаю эти результаты обоснованием для GNN, более крупного
encoder или заявления «модель вообще игнорирует evidence». Контроль показывает
optimization drift; blank probe показывает broad conditioning; локальные пары
выявляют слабый relation-conditioned сигнал.

Следующий наиболее информативный packet: **natural evidence-counterfactual
supervision с неизменной архитектурой**, прежде чем менять encoder. Небольшой
набор independently reviewed естественных train claims; тот же ответ, исходный
и контролируемо изменённый evidence, оба конкурирующих значения присутствуют.
Control — duplicate raw CE, treatment — CE на проверенных paired factual worlds;
одинаковые replay, exposures, steps и B7 initialization. Только target spans,
обусловленные изменением evidence; не label flipping всех ответов документа.
Evaluation на source-disjoint natural factual pairs и сохранённой natural/localization
проверке. Основной criterion — claim/slot gap и gold-peak alignment при retention,
а не broad blank response или одним response F1. Сначала annotation feasibility
audit и профиль, затем бюджет не более одного такого matched seed/30 GPU-min.
**Этот следующий fit не запущен и требует нового preregistered packet.**

Если такой сигнал переносится, но response max всё ещё выбирает punctuation/filler,
тогда обоснована архитектурная ablation evidence-conditioned claim head против
max-token. Сохранять joint full context/coreference; не повторять неудачный R9
independent-passage pooling и не смешивать новую head и supervision в одном control.
Это проверяемое направление, не claim of novelty.

## Инженерия и воспроизведение

Оба fits завершены по фиксированному расписанию; optimizer-resume не нужен.
Checkpoints содержат final weights/config/tokenizer, не выбранный на inner best epoch:

- B7: `models/phase2b/span_inner_s42/best`, SHA256
  `7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4`.
- R10-A: `models/phase2b_r10/R10As42/final`, SHA256
  `141ecba5b3f1dad5c2a140024127f4d970cd0b3fbeef0981c35ef451e164ab7c`.
- R10-B: `models/phase2b_r10/R10Bs42/final`, SHA256
  `9acebe16b8f995b457de11094e5cc0591b4b8ff3d1ae4112ad8576a41a1423bd`.

Fit times A492.52s/B487.44s; peak allocated2.952GiB/reserved3.123GiB;
peak process RSS2.51/2.65GiB. Final table gives resident scoring timings; they exclude
model loading/tokenization. Whole evaluation jobs B7/A/B26.67/96.29/96.28s include
loading/encoding/guards. B7 natural/rotations/R6 predictions reused only after hash
verification; R1/blank freshly scored. Historical B7 natural runtime26.01s,
24.68 responses/s, peak0.644GiB, **not a new R10 timing run**.

Verification: offline **246 passed/8 skipped**; opt-in local research
**253 passed/1 skipped** (official real-dataset regression deliberately disabled);
analysis sanity **8 passed**. Two offline warnings concern an unwritable pytest
cache, not test failures. Research/analysis use a writable local cache.
Ruff and both `pip check` passed. CI file exists and is tracked, Python3.11/3.12,
offline tests; no remote CI success claimed.

One post-hoc display-pack correction: an ad hoc Windows default-encoding read
corrupted several non-ASCII excerpts in `review_cases.jsonl`. Verification caught it;
the invalid pack and failed log are retained. Correct pack:
`reports/phase2b_r10/review_cases_utf8.jsonl`, reproduced by `audit_cases.py`.
Predictions, records, metrics and preregistered analysis all used explicit UTF-8;
none were changed. The Markdown detail-table header uses spelled-out absolute
changes to avoid embedded pipe delimiters; numeric values are unchanged.

New/updated source files: `research/permutation.py`, `scripts/research_r10.py`,
`tests/test_permutation.py`, this report/proposal, PROGRESS, DECISIONS,
EVALUATION_PROTOCOL, RESEARCH_NOTES, `.gitignore`. Local execution/analysis helpers
under `models/phase2b_r10/*.py` are included in the source snapshot, so ignored
helpers are recoverable. Texts, weights, caches and predictions remain uncommitted.
No push/publication/release. Historical136 R9 manifest entries and checkpoint hashes
are verified again; old source/dependency guards are not bypassed.

Final source: `research_snapshots/R10_source_20261025.zip` with member hashes;
artifact/checkpoint manifest: `models/phase2b_r10/R10_artifact_manifest.json`.
`freeze.json`, `inputs.json`, `analysis_code_freeze.json`, training configs, batch
logs, queue commands, predictions and gate are retained locally. Original source
snapshot used during fitting is separately referenced by `executed_source.json`.

Read-only verification from repository root (PowerShell):

```powershell
$env:PYTHONPATH = "src;scripts"
.venv-research\Scripts\python.exe models\phase2b_r10\verify.py
.venv-research\Scripts\python.exe models\phase2b_r10\audit_cases.py
```

Actual experiment commands, **not to rerun into completed directories**:

```powershell
$env:PYTHONPATH = "src;scripts"
.venv-research\Scripts\python.exe models\phase2b_r10\run_queue.py profile
.venv-research\Scripts\python.exe models\phase2b_r10\run_queue.py fit
.venv-research\Scripts\python.exe models\phase2b_r10\run_queue.py evaluate
.venv-research\Scripts\python.exe models\phase2b_r10\analyze.py
.venv-research\Scripts\python.exe models\phase2b_r10\summarize.py
```

Replay requires a separate checkout of the archived executed source and the same
hash-verified frozen inputs/historical B7 artifacts, with empty new R10 output
directories. Do not delete or overwrite current results, regenerate old freezes,
or skip source/dependency checks. The initial allowlist extraction is recorded in
`prepare.py`; final screen can be reproduced exactly with `audit_cases.py`.

Checks:

```powershell
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m pytest -q -o cache_dir=models/phase2b_r10/check_cache
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m pip check
.venv-research\Scripts\python.exe -m pip check
$env:GROUNDCHECK_RESEARCH_MODELS = "1"
.venv-research\Scripts\python.exe -m pytest -q -o cache_dir=models/phase2b_r10/check_research_cache
$env:PYTHONPATH = "src;scripts;models/phase2b_r10"
.venv-research\Scripts\python.exe -m pytest models/phase2b_r10/test_analysis.py -q
```

## Полные измеренные таблицы

Ниже автоматически перенесены значения из `analysis.json`. Все absolute/paired
intervals, length/generator slices и source denominators сохранены в JSON; таблицы
не заменяют полные diagnostics. Synthetic и natural результаты не объединяются.


## Natural original order: 642 responses / 107 reused inner sources

| Method | Accuracy | Macro-F1 | Positive P | Positive R | Positive F1 | AUROC | AP | FPR | FNR | CM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B7 | 0.8162 | 0.7956 | 0.6838 | 0.7843 | 0.7306 | 0.8759 | 0.7897 | 0.1689 | 0.2157 | [[364, 74], [44, 160]] |
| R10As42 | 0.8162 | 0.7875 | 0.7129 | 0.7059 | 0.7094 | 0.8701 | 0.7792 | 0.1324 | 0.2941 | [[380, 58], [60, 144]] |
| R10Bs42 | 0.8131 | 0.7844 | 0.7059 | 0.7059 | 0.7059 | 0.8735 | 0.7831 | 0.1370 | 0.2941 | [[378, 60], [60, 144]] |

| Method | Token F1 | Character F1 | Exact-span F1 | Short TP/19 | Short gold-hit/19 | Detected-positive peak on gold |
| --- | --- | --- | --- | --- | --- | --- |
| B7 | 0.5870 | 0.5912 | 0.0379 | 14 | 8 | {'n': 160, 'on_gold': 118} |
| R10As42 | 0.5769 | 0.5787 | 0.0389 | 11 | 6 | {'n': 144, 'on_gold': 107} |
| R10Bs42 | 0.5796 | 0.5803 | 0.0352 | 11 | 6 | {'n': 144, 'on_gold': 104} |

| Method | Macro-F1 95% CI | Token F1 95% CI |
| --- | --- | --- |
| B7 | [0.7633, 0.8273] | [0.5105, 0.6557] |
| R10As42 | [0.7506, 0.8228] | [0.4980, 0.6514] |
| R10Bs42 | [0.7485, 0.8199] | [0.4975, 0.6516] |

| Difference | Macro-F1 delta 95% CI | Token F1 delta 95% CI |
| --- | --- | --- |
| R10Bs42 - R10As42 | [-0.0143, 0.0077] | [-0.0046, 0.0114] |
| R10Bs42 - B7 | [-0.0356, 0.0121] | [-0.0216, 0.0056] |

## Permissible-order diagnostic: 204 responses / 34 sources

| Method | Union flips /204 | Rate 95% CI | Mean absolute score change | Mean absolute token change | Raw MF1 | Rotation1 MF1 | Rotation2 MF1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B7 | 18 | [0.0490, 0.1324] | 0.0420 | 0.0166 | 0.7656 | 0.7676 | 0.7742 |
| R10As42 | 13 | [0.0294, 0.1029] | 0.0447 | 0.0154 | 0.7539 | 0.7783 | 0.7662 |
| R10Bs42 | 10 | [0.0147, 0.0882] | 0.0440 | 0.0144 | 0.7539 | 0.7783 | 0.7613 |

| Difference | Flip-rate delta 95% CI | Mean score-change delta 95% CI |
| --- | --- | --- |
| R10Bs42-R10As42 | [-0.0343, 0.0049] | [-0.0048, 0.0036] |
| R10Bs42-B7 | [-0.0882, 0.0049] | [-0.0034, 0.0073] |

## Existing R6 controlled facts: 384 pairs / 16 parent sources

| Method | Macro-F1 | Positive F1 | Supported FPR | Contradiction recall | Token F1 | Exact-span F1 | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B7 | 0.4399 | 0.2605 | 0.1823 | 0.1771 | 0.0024 | 0.0000 | 0.5213 | 0.5151 |
| R10As42 | 0.4426 | 0.2597 | 0.1693 | 0.1745 | 0.0005 | 0.0000 | 0.5191 | 0.5102 |
| R10Bs42 | 0.4277 | 0.2280 | 0.1536 | 0.1484 | 0.0000 | 0.0000 | 0.5187 | 0.5101 |

| Method | Claim gap [95% CI] | Slot gap | Response gap | Claim rank fraction | Pair accuracy |
| --- | --- | --- | --- | --- | --- |
| B7 | 0.0074 [0.0015, 0.0137] | 0.0081 | 0.0067 | 0.6354 | 0.0182 |
| R10As42 | 0.0053 [0.0004, 0.0107] | 0.0057 | 0.0054 | 0.6120 | 0.0052 |
| R10Bs42 | 0.0037 [-0.0003, 0.0079] | 0.0044 | 0.0036 | 0.6042 | 0.0052 |

| Difference | Claim-gap delta 95% CI | Ranking delta 95% CI |
| --- | --- | --- |
| R10Bs42-R10As42 | [-0.0042, 0.0010] | [-0.0260, 0.0104] |
| R10Bs42-B7 | [-0.0073, -0.0000] | [-0.0729, 0.0156] |

| Category | Pairs / parents | B7 claim gap / rank | A claim gap / rank | B claim gap / rank |
| --- | --- | --- | --- | --- |
| entity_swap | 64 / 16 | 0.0220 / 0.7656 | 0.0181 / 0.7188 | 0.0119 / 0.7344 |
| long_short | 64 / 16 | 0.0026 / 0.6250 | -0.0007 / 0.5781 | 0.0001 / 0.5156 |
| property_swap | 64 / 16 | 0.0086 / 0.6562 | 0.0062 / 0.6250 | 0.0046 / 0.6875 |
| prose_order | 64 / 16 | 0.0051 / 0.5469 | 0.0034 / 0.5469 | 0.0005 / 0.4219 |
| shared_value | 64 / 16 | 0.0069 / 0.6406 | 0.0027 / 0.6562 | 0.0018 / 0.7031 |
| similar_name | 64 / 16 | -0.0010 / 0.5781 | 0.0021 / 0.5469 | 0.0034 / 0.5625 |
| long | 64 / 16 | 0.0026 / 0.6250 | -0.0007 / 0.5781 | 0.0001 / 0.5156 |
| short | 320 / 16 | 0.0083 / 0.6375 | 0.0065 / 0.6188 | 0.0044 / 0.6219 |

## Existing R1 controlled facts: 75 pairs / 15 parent sources

| Method | Macro-F1 | Positive F1 | Supported FPR | Contradiction recall | Token F1 | Exact-span F1 | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B7 | 0.3680 | 0.0941 | 0.0800 | 0.0533 | 0.0492 | 0.0000 | 0.5204 | 0.5302 |
| R10As42 | 0.3658 | 0.0741 | 0.0400 | 0.0400 | 0.0171 | 0.0000 | 0.5067 | 0.5150 |
| R10Bs42 | 0.3693 | 0.0750 | 0.0267 | 0.0400 | 0.0000 | 0.0000 | 0.4949 | 0.5100 |

| Method | Claim gap [95% CI] | Slot gap | Response gap | Claim rank fraction | Pair accuracy |
| --- | --- | --- | --- | --- | --- |
| B7 | 0.0105 [0.0052, 0.0162] | 0.0115 | 0.0034 | 0.7600 | 0.0000 |
| R10As42 | 0.0048 [0.0023, 0.0074] | 0.0055 | 0.0004 | 0.7333 | 0.0000 |
| R10Bs42 | 0.0042 [0.0019, 0.0071] | 0.0031 | -0.0004 | 0.6933 | 0.0133 |

| Difference | Claim-gap delta 95% CI | Ranking delta 95% CI |
| --- | --- | --- |
| R10Bs42-R10As42 | [-0.0035, 0.0023] | [-0.1333, 0.0533] |
| R10Bs42-B7 | [-0.0111, -0.0020] | [-0.1467, 0.0133] |

| Category | Pairs / parents | B7 claim gap / rank | A claim gap / rank | B claim gap / rank |
| --- | --- | --- | --- | --- |
| attribution | 15 / 3 | 0.0089 / 0.4667 | 0.0055 / 0.6000 | 0.0083 / 0.4667 |
| date | 15 / 3 | 0.0200 / 0.9333 | 0.0040 / 0.9333 | 0.0061 / 0.8667 |
| negation | 15 / 3 | 0.0080 / 0.8000 | 0.0080 / 0.7333 | 0.0019 / 0.6000 |
| quantity | 15 / 3 | 0.0087 / 0.7333 | 0.0038 / 0.7333 | 0.0025 / 0.8000 |
| unit | 15 / 3 | 0.0068 / 0.8667 | 0.0029 / 0.6667 | 0.0021 / 0.7333 |
| long | 60 / 15 | 0.0090 / 0.7333 | 0.0032 / 0.6833 | 0.0036 / 0.6833 |
| short | 15 / 15 | 0.0164 / 0.8667 | 0.0115 / 0.9333 | 0.0065 / 0.7333 |

## Broad empty-evidence probe (no new labels)

96 responses / 16 sources; original negatives=66.

| Method | Mean original-negative risk shift | 95% CI |
| --- | --- | --- |
| B7 | 0.7071 | [0.5725, 0.8100] |
| R10As42 | 0.7450 | [0.6003, 0.8575] |
| R10Bs42 | 0.7622 | [0.6128, 0.8733] |

## Runtime / memory

| Fit | Seconds | Updates | Peak allocated GiB | Peak reserved GiB |
| --- | --- | --- | --- | --- |
| R10As42 | 492.5208 | 642 | 2.9524 | 3.1230 |
| R10Bs42 | 487.4414 | 642 | 2.9524 | 3.1230 |

| Method | Suite | Resident scoring seconds | Responses/s | Forward p50 ms | Forward p95 ms | Peak allocated GiB |
| --- | --- | --- | --- | --- | --- | --- |
| B7 | r1 | 5.6363 | 26.6133 | 34.9380 | 45.3709 | 0.6181 |
| B7 | blank | 3.5394 | 27.1229 | 35.2071 | 42.0674 | 0.6237 |
| R10As42 | natural | 25.5344 | 25.1425 | 36.3086 | 53.7064 | 0.6435 |
| R10As42 | rotate1 | 7.9782 | 25.5697 | 36.4298 | 50.3240 | 0.6370 |
| R10As42 | rotate2 | 8.1131 | 25.1444 | 36.4440 | 52.6916 | 0.6370 |
| R10As42 | r6 | 26.9746 | 28.4712 | 33.6334 | 41.0865 | 0.6217 |
| R10As42 | r1 | 5.4557 | 27.4944 | 34.9556 | 43.1950 | 0.6181 |
| R10As42 | blank | 3.7003 | 25.9438 | 36.8947 | 48.2232 | 0.6237 |
| R10Bs42 | natural | 25.1906 | 25.4857 | 35.9841 | 53.2481 | 0.6435 |
| R10Bs42 | rotate1 | 8.0444 | 25.3593 | 36.9992 | 47.9760 | 0.6370 |
| R10Bs42 | rotate2 | 7.9233 | 25.7469 | 36.3376 | 48.2558 | 0.6370 |
| R10Bs42 | r6 | 27.4438 | 27.9845 | 33.6634 | 44.7956 | 0.6217 |
| R10Bs42 | r1 | 5.4901 | 27.3219 | 35.0897 | 43.8577 | 0.6181 |
| R10Bs42 | blank | 3.5709 | 26.8839 | 35.9109 | 44.2062 | 0.6237 |

## Frozen gates

| Gate | Passed |
| --- | --- |
| invariance_nonzero_control | True |
| invariance_25pct_reduction | False |
| invariance_paired_upper_below_zero | False |
| invariance_not_worse_B7 | True |
| eligible_worst_order_retained | True |
| natural_macro_noninferiority_vs_R10As42 | False |
| natural_recall_retained_vs_R10As42 | True |
| natural_token_f1_retained_vs_R10As42 | True |
| short_no_individual_hit_lost_vs_R10As42 | True |
| natural_macro_noninferiority_vs_B7 | False |
| natural_recall_retained_vs_B7 | False |
| natural_token_f1_retained_vs_B7 | False |
| short_no_individual_hit_lost_vs_B7 | False |
| natural_TPR_TNR_floor | True |
| r6_claim_gap_floor | False |
| r6_claim_rank_floor | True |
| r6_claim_gap_noninferiority | True |
| r6_claim_rank_noninferiority | True |
| r1_claim_gap_floor | False |
| r1_claim_rank_floor | True |
| r1_claim_gap_noninferiority | True |
| r1_claim_rank_noninferiority | False |
| blank_shift_floor | True |
| blank_noninferiority | True |

Extra seeds authorized: False
