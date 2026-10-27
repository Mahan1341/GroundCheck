# R8 — Natural evidence graph extraction (2026-10-24)

**Пакет выполнен; результат отрицательный для текущего extractor.** Кандидаты
появились в137/144 held-out ответах, но end-to-end structural coverage остался
**0/144**. Flat lookup и graph совпадают с neural baseline. Sentence retrieval
повышает recall ценой новых FP. Не обучаем graph-aware detector на этих proposals.
`Пакет ручного review` (historical local artifact; not distributed):48 примеров/48 sources, ни одного
утверждённого gold graph. Требование reviewed precision остаётся pending;
автоматический coverage gate уже failed. Исследование не завершено, release нет.

## Данные и freeze

Все источники взяты из прежнего R6 natural inner snapshot, а не из outer.
48 sources выбраны первым48 SHA256(R8:42:source_id), без labels/predictions;
известный duplicate15181 исключён заранее. First24 — extractor-development,
last24 — held-out evaluation. По144 ответа: development95/49, evaluation105/39
supported/hallucinated. 24+24 source groups не пересекаются и не входят в
supervised fitting frozen R6-B. Query types включают процедуры, определения,
здоровье, цены, историю/права и бытовые инструкции; hash sample не stratified.

Source IDs сохранены до просмотра development prose в models/phase2b_r8/source_split.json.
Правила редактировались только на development. EXPERIMENT_R8_PLAN.md и source,
resource, input hashes frozen до новых model predictions. После evaluation не
меняли parser, retrieval, verdict, thresholds или набор. Это held-out только
относительно разработки R8 extractor: весь inner set исторически изучен, исходный
порог выбран раньше внутри train. Не называем его untouched external evaluation.

Использованы R6-B42 weights SHA256
`86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e`,
FP32 batch1 TF32off, response >=.37612172961235046, token .5. No fitting.
CPU preflight: max857 development/889 evaluation tokens, budget2048,
answer/evidence coverage1, truncation0. Все восемь sequential stages завершены.
Baseline token probabilities точно воспроизвели R6 на тех же IDs.

## Почему R7 не работал и что изменено

R7 full-match grammar принимала четыре scalar clause формы и шесть unit strings.
В natural sources: passage/list headers, imperative instructions, copulas и
verb relations, non-numeric complements, approximate quantities/ranges,
conditional scope, pronouns и implicit arguments. Поверхностное совпадение числа
не восстанавливает эти связи. Глобальный numeric-unknown veto R7 также не подходит
для документов, где часть чисел относится к другим фактам.

Расширен существующий evidence_graph.py, прежние R7 entry points сохранены;
для старого freeze нужен старый snapshot. Новый режим:

1. Local Punkt по paragraphs с сохранением глобальных character offsets.
2. Treebank token spans и небольшой English averaged-perceptron POS tagger.
   NN/NNP/pronoun mentions плюс subject interval — предложения сущностей, не NER gold.
3. First-verb/auxiliary-chain heuristic → subject / normalized predicate / complete
   complement. Один candidate на clause; ничего не генерируется. Units сохраняются
   в value, поле unit=text, числовой canonicalization нет.
4. Literal tense/modality/condition/negation; complex/implicit/coreferential,
   embedded/range/approximate cases обычно остаются candidate/unknown.
5. Explicit complete known-as/called/referred-to-as evidence clauses → alias edges.
   Claim aliases не являются достоверным evidence. Не делаем fuzzy/coreference joins.
6. Provenance document/passage/clause/subject/predicate/value offsets; confidence
   rule_high_unreviewed/provisional — эвристические статусы, НЕ probabilities.

Использованы прежний NLTK/Punkt и новый архив POS всего1,539,115 bytes, никаких
новых Python dependencies. Revision nltk_data4f15a3d89eefe9748ec1c05be495d91289197155,
SHA2566025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b.
Файлы проверяются по configs/r8-nlp.json; inference не скачивает ресурсы.
Версия NLTK и все зависимости входят в freeze. Extraction работает на CPU.

## Чистое сравнение A/B/C/D

A — полный frozen neural input. B — тот же набор candidate claim intervals,
fixed top2 Jaccard evidence sentences + same-passage neighbors. Union контекста
проверяется дополнительным neural pass с полным answer; остальные token scores
сохраняются от baseline. Question не используется как evidence, learned IDF нет.

C flat и D graph используют **побитово одинаковые extracted graphs**, confidence
filters, qualifiers, open-world verdict policy и override rules. Только D
разрешает traversal explicit evidence alias edges; C сопоставляет entity strings.
Support требует равного полного complement, polarity и qualifiers. Contradiction
допускается при явной противоположной polarity того же немодального утверждения;
разные natural values сами по себе остаются unknown. Не предполагаем, что любой
natural predicate single-valued. Это сильно ограничивает numerical error detection.

Resolved assertion меняет только свой clause risk на0/1; unknown/ambiguous
сохраняет baseline. Это экспериментальные symbolic scores, не calibrated confidence.
Oracle facts и hallucination spans не подавались в extraction/retrieval/inference.

## Измеренные результаты

2000 paired source bootstrap resamples,seed42,24 parents в каждой части.
Варианты/шесть generated answers одного source группируются вместе. CI условны
на checkpoint/выбранные sources/rules и не корректируют историю model selection.
CM rows=true/columns=pred [0,1]. AP — Average Precision, не trapezoidal PR-AUC.

### evaluation

| Метод | Macro-F1 [95% CI] | Positive P / R / F1 | FPR | AUROC / AP | CM [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---|
| baseline | 0.7645 [0.6647, 0.8571] | 0.6500 / 0.6667 / 0.6582 | 0.1333 | 0.8552 / 0.7134 | [[91, 14], [13, 26]] |
| sentence | 0.7152 [0.6133, 0.8125] | 0.5167 / 0.7949 / 0.6263 | 0.2762 | 0.8176 / 0.6337 | [[76, 29], [8, 31]] |
| flat | 0.7645 [0.6647, 0.8571] | 0.6500 / 0.6667 / 0.6582 | 0.1333 | 0.8552 / 0.7134 | [[91, 14], [13, 26]] |
| graph | 0.7645 [0.6647, 0.8571] | 0.6500 / 0.6667 / 0.6582 | 0.1333 | 0.8552 / 0.7134 | [[91, 14], [13, 26]] |

| Метод | Token / character / exact-span F1 | Short TP / gold hit / n | ms/ответ | ответов/s | Peak VRAM / process RSS GiB |
|---|---:|---:|---:|---:|---:|
| baseline | 0.5545 / 0.5735 / 0.0335 | 1 / 0 / 2 | 41.28 | 24.22 | 0.640 / 1.438 |
| sentence | 0.5327 / 0.5489 / 0.0159 | 1 / 1 / 2 | 97.90 | 10.21 | 0.641 / 1.437 |
| flat | 0.5545 / 0.5735 / 0.0335 | 1 / 0 / 2 | 61.79 | 16.18 | 0.640 / 1.437 |
| graph | 0.5545 / 0.5735 / 0.0335 | 1 / 0 / 2 | 60.68 | 16.48 | 0.640 / 1.438 |

| Разность с baseline | Δ macro-F1 [95% paired CI] |
|---|---:|
| sentence | -0.0493 [-0.1224, 0.0178] |
| flat | +0.0000 [0.0000, 0.0000] |
| graph | +0.0000 [0.0000, 0.0000] |


### development

| Метод | Macro-F1 [95% CI] | Positive P / R / F1 | FPR | AUROC / AP | CM [[TN,FP],[FN,TP]] |
|---|---:|---:|---:|---:|---|
| baseline | 0.7481 [0.6625, 0.8159] | 0.6271 / 0.7551 / 0.6852 | 0.2316 | 0.8455 / 0.7319 | [[73, 22], [12, 37]] |
| sentence | 0.7298 [0.6411, 0.8084] | 0.5753 / 0.8571 / 0.6885 | 0.3263 | 0.8258 / 0.7099 | [[64, 31], [7, 42]] |
| flat | 0.7481 [0.6625, 0.8159] | 0.6271 / 0.7551 / 0.6852 | 0.2316 | 0.8455 / 0.7319 | [[73, 22], [12, 37]] |
| graph | 0.7481 [0.6625, 0.8159] | 0.6271 / 0.7551 / 0.6852 | 0.2316 | 0.8455 / 0.7319 | [[73, 22], [12, 37]] |

| Метод | Token / character / exact-span F1 | Short TP / gold hit / n | ms/ответ | ответов/s | Peak VRAM / process RSS GiB |
|---|---:|---:|---:|---:|---:|
| baseline | 0.5887 / 0.5863 / 0.0532 | 3 / 3 / 4 | 41.09 | 24.34 | 0.638 / 1.438 |
| sentence | 0.5622 / 0.5580 / 0.0310 | 4 / 4 / 4 | 100.12 | 9.99 | 0.638 / 1.438 |
| flat | 0.5887 / 0.5863 / 0.0532 | 3 / 3 / 4 | 63.94 | 15.64 | 0.638 / 1.438 |
| graph | 0.5887 / 0.5863 / 0.0532 | 3 / 3 / 4 | 63.03 | 15.86 | 0.638 / 1.438 |

| Разность с baseline | Δ macro-F1 [95% paired CI] |
|---|---:|
| sentence | -0.0183 [-0.0704, 0.0329] |
| flat | +0.0000 [0.0000, 0.0000] |
| graph | +0.0000 [0.0000, 0.0000] |


## Coverage: кандидаты не равны проверенным фактам

| Показатель | Development | Held-out evaluation |
|---|---:|---:|
| Ответы с хотя бы одним candidate |136/144 (94.4%) |137/144 (95.1%) |
| Ответы с eligible rule-high candidate |26/144 (18.1%) |36/144 (25.0%) |
| Candidate claims / segmented clauses |819/1236 |805/1229 |
| Eligible claims / candidates |45/819 |50/805 |
| Numeric candidates |271/819 |160/805 |
| Sources с evidence candidates |24/24 |23/24 |
| Sources с eligible evidence relation |11/24 |13/24 |
| Unique-source evidence candidates / clauses |252/316 |256/325 |
| Eligible evidence relations |25/252 |24/256 |
| Numeric evidence candidates |70/252 |50/256 |
| End-to-end structural response coverage |5/144 (3.47%),3 sources |**0/144,0 sources** |
| Structural supported / contradicted assertions |7 /0 |0 /0 |
| Extracted explicit alias edges |0 |0 |

Evidence counts здесь deduplicated по source; один и тот же документ не считается
шесть раз. Все candidate counts — proposal yield, НЕ extraction recall.
POS NN mentions не гарантируют реальные entity boundaries. Из50 eligible held-out
claims49 не нашли exact entity/property key,1 abstained из-за неоднозначного
extracted evidence. Ещё755/805 candidates сразу unknown на claim extraction.
Natural ambiguity1/805, unknown804/805; unresolved clauses без candidate также
учитываются отдельно:1179/1229 clauses не имеют eligible assertion.

Flat/graph predictions идентичны на обеих частях. На development изменились
низкие token probabilities пяти ответов, но не response решения и не token/span
метрики при .5. На evaluation вообще не изменился ни один token score.
**Нет alias-dependent cases: этот набор не позволяет измерить пользу graph traversal.**
Нельзя заключать, что любые графы бесполезны, из эквивалентности неактивных режимов.

Transfer proxy:109/144 held-out ответов из24 sources содержат хотя бы один predicate
stem, отсутствовавший в development. Baseline/flat/graph macro-F1=.738227,
sentence=.725594,32 positives. Structural coverage и здесь0. Это перенос за
словарь извлечённых predicate strings, не независимо размеченные relation families:
часть новых strings — ошибки POS (`/`, `*` и др.). Полноценный semantic transfer
на этом основании не доказан. Небольшой local-NLP test дополнительно проверяет
новые простые copular/verb формы, но это только engineering regression.

## Ошибки и annotation quality

В docs/NATURAL_GRAPH_REVIEW.md48 автоматически выбранных proposals, один на source,
с question, исходными claim/evidence clauses, document/passage provenance,
subject/predicate/value offsets, qualifiers, confidence, suspected issues и
незаполненными reviewer fields. Выбор: resolved match, затем eligible candidate,
lexical proximity и deterministic tie; без RAGTruth gold/model correctness.
Machine graph не объявляется human annotation. Такая приоритетная выборка также
не является случайной оценкой precision всей популяции.

Реальные предварительные замечания ассистента (не независимый gold audit):

- Review12,source15150: `Add sliced onions ...` → entity=`Add`, predicate=`slice`.
  Это императив с adjectival participle, а не такое отношение. Машинный support
  возникает при совпадении той же неверной структуры в evidence. Истинность всей
  скопированной инструкции не доказывает корректность triple extraction.
- Review13,source15191: `Place trimmed Brussels sprouts ...` → entity=`Place`,
  predicate=`trim`. Та же POS/argument-boundary ошибка. Добавленный до freeze
  lowercase imperative guard оказался недостаточен; не исправляли после inference.
- Review7/8/14: desired/baking/nursing могут ошибочно стать predicate внутри
  noun phrase; rule_high_unreviewed не является надёжным confidence estimator.
- Review32,source14361: формула инфляции дала predicate=`/`; units/formulas требуют
  специального conservative parser или abstention, а не verb-stem matching.
- Review40,source15446: корректно выглядящее `Figs are the fruit ...` не сопоставлено
  с evidence, где добавлено relative clause `which ...`; candidate evidence
  блокирует verdict. Здесь strict complete-complement matching теряет paraphrase.
- Review25,source12215: `has` → `ha`, `have` → `have`: Porter stemming не является
  полноценной lemmatization. Нужна морфология, а не новые detector weights.
- Review45,source15523: attribution `The Americans claimed that ...` превращается
  в embedded candidate. Нельзя выбросить attribution и считать вложенное утверждение
  фактом evidence; нужны отдельно scope/attribution и role spans.
- Review35,source14419: price/coupon fragments без finite verb не дают evidence
  relation. Это measured no-candidate source, не доказанное insufficient evidence.

Точное восстановление offsets для48 proposals проверено автоматически против
исходных строк. Это проверка координат, не семантическая валидация извлечения.
Ни один случай не утверждён пользователем: reviewed0. Extraction/linking P/R
оставлены null. Для recall пользователь должен дополнительно разметить пропущенные
relations в полном источнике; принятие предложенных triples само по себе recall
не определяет. Scope/coreference/implicit relations остаются главной открытой
частью audit. До появления alias cases нельзя оценить entity-linking accuracy.

False symbolic overrides:0 новых response-level ошибок и0 автоматических
supported-overlap-with-RAGTruth-span flags. На held-out при0 overrides это
**не оценка высокой точности**. На development7 support assertions не пересекали
hallucination spans; отсутствие RAGTruth error span не является human evidence
entailment label. Поля confidence и отсутствие flags не разрешают production use.

Sentence retrieval: на held-out15 новых ошибок и5 исправленных; Δmacro-F1−.04927,
95%CI[−.12241,+.01779], поэтому убедительная деградация macro-F1 не установлена.
Но FPR вырос на.14286, paired95%CI[+.05882,+.25234]. Mean selected original evidence
character fraction .7223 (включает zero-selection fallback), .7407 на development.
Изменения связаны с отбором/форматированием контекста; полнота найденной поддержки
не размечена, retrieval Recall@k не вычисляется.

Примеры сохранены локально в reports/phase2b_r8/sentence_error_examples.json:
response12498/source14401,gold0: risk .00265→.99666 — новый FP после выбора62.7%
исходных evidence characters. Response15316/source15473,gold1: .30870→.87864 —
исправленный FN при52.1% characters. Сам факт смены решения не устанавливает,
какая конкретная удалённая sentence причинно ответственна; нужна отдельная ablation.

Custom localization scorer прежний: token any gold-char overlap, nonspace char,
exact trimmed/merged span, threshold.5; не официальный RAGTruth scorer.
Short slice >=500 chars, gold nonspace fraction<=5%: только2 held-out примера
и4 development. У sentence held-out gold hit0→1/2, response TP1/2 не меняется;
такая малая выборка не доказывает улучшение localization. Aggregate token/span F1
снижаются. В short bootstrap1759/2000 held-out draws имеют непустой denominator;
остальные не выдаются за нулевой/perfect score. Полные CIs — analysis JSON.

## Gate и следующий эксперимент

Заранее: structural coverage>=5% ответов из>=4 sources; graph−baseline macro-F1
>=−.01 с lower95%>−.03, recall loss<=.02, short gold hit loss<=1. Reviewed structural
precision>=.95 на>=20 независимо утверждённых decisions из>=8 sources.
Итог **failed_coverage_and_pending_review**. Retention checks true лишь потому,
что graph ничего не изменил. Gate reviewed precision не прошёл и не провалился:
данных для него пока нет. Никаких rescue thresholds или дополнительных seeds.

Следующий наиболее информативный архитектурный эксперимент — заменить только
first-verb POS heuristic на небольшой dependency-based extractive parser с
subject/predicate/complement spans, полноценной lemmatization и явными qualifier
roles. Контроль — этот frozen POS extractor; C/D verdict, ModernBERT и aggregation
сохранить. Сначала утвердить review pack и создать новую закрытую source-group
evaluation, поскольку R8 held-out теперь просмотрен. Измерить argument-boundary
и scope errors, exact relation precision, abstention и end-to-end coverage до
любого обучения detector. Не лечить нулевую coverage ослаблением support rules.
Новая модель/parser потребует pinned revision и CPU profiling; сейчас не запускалась.

Это известная extraction/dependency-parsing постановка, не заявленная новизна.
Нет оснований сейчас обучать GNN или graph-aware hallucination head на ошибочных
POS pseudo-labels. Пользовательская проверка остаётся следующим конкретным шагом.

## Проверки, ресурсы, сохранение

- 229 passed/1 skipped в research env (69.17s);222 passed/8 skipped offline (54.29s).
  GROUNDCHECK_RAGTRUTH_DIR unset; legacy majority/dev regression не запускалась.
- 26 новых tests, включая optional pinned-NLP test: shared C/D policy, absent/different
  values unknown, negation/modality/qualifiers, explicit alias ambiguity, local
  overrides, недоверие answer aliases, offsets, implicit subjects/refusal handling.
- Ruff passed; pip check в обеих venv: No broken requirements found. Dependencies
  не менялись. CI3.11/3.12 offline остаётся; remote CI success не проверялся.
- Resident-model pipeline sum76.30s; extraction/tagger CPU входит в latency, load/disk
  и bootstrap исключены. Peak VRAM .638–.641GiB allocated; process RSS peak~1.44GiB.
  Это один timing trial; нет заявлений о точной сравнительной производительности.
- Старые85 R7 artifacts и checkpoint сохранены, hashes проверены. Новые predictions,
  graphs, diagnostics, configs, split, freeze, logs и WORK_STATE — phase2b_r8.
  Source snapshot research_snapshots/R8_source_20261024.zip + member hashes.
  Dataset text/review pack не входит в source ZIP; отдельный локальный
  models/phase2b_r8/review_pack.zip сохраняет proposals, review document и audit.
  Инвентарь models/phase2b_r8/R8_artifact_manifest.json. Никаких commits/push/release.

Основные изменения: research/evidence_graph.py (R8 extension), research/natural_nlp.py,
scripts/research_r8.py, configs/r8-nlp.json, tests/test_natural_graph.py,
EXPERIMENT_R8_PLAN.md, .gitignore, docs/NATURAL_GRAPH_REVIEW.md,
docs/PHASE2B_R8_RESULTS.md, PROGRESS.md, DECISIONS.md, EVALUATION_PROTOCOL.md,
RESEARCH_NOTES.md. Helper scripts models/phase2b_r8/*.py включены в source snapshot,
хотя сама models/ папка ignored. Review document и text-bearing error examples
исключены из Git; approved annotations в этом пакете нет.

## Воспроизведение

В текущей копии безопасно повторять только проверки, не перезаписывая outputs:

```powershell
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m pytest -q
$env:GROUNDCHECK_RESEARCH_MODELS='1'
.venv-research\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pip check
.venv-research\Scripts\python.exe -m pip check
```

Полный replay: отдельная восстановленная Windows рабочая копия R8 source ZIP,
прежние local R6/R7 manifests/checkpoint/inner records/raw hashes, frozen research
venv. Исключить прежние R8 reports/output files; source_split.json уже включён в
source snapshot с исходными bytes (models/phase2b_r8/source_split.json). Source archive не содержит
weights/raw text. Ресурсный manifest использует исходные Windows relative paths.

```powershell
$env:PYTHONPATH="$PWD\src"
$py='.venv-research\Scripts\python.exe'
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
& $py models/phase2b_r8/fetch_nlp.py
& $py scripts/research_r8.py freeze
& $py models/phase2b_r8/preflight.py
& $py models/phase2b_r8/run_packet.py
& $py models/phase2b_r8/analyze.py
& $py models/phase2b_r8/review_pack.py
& $py models/phase2b_r8/gate_tables.py
& $py models/phase2b_r8/diagnostics.py
```

Существующие exclusive-write guards намеренно запрещают повтор поверх результата.
Не обходить source/dependency mismatches старых freezes. Обучений для resume нет;
все inference stages завершены. Остановились перед новым обучением, research активен.
