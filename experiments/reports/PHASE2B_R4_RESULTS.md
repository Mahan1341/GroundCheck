# R4 — Numerical Grounding & Local Claims

2026-10-24. Оба inference-эксперимента завершены; исследование продолжается. Обучений, outer calibration/dev/test и release в этом пакете не было.

## Главный результат

Симметричная нормализация помогает уже обученному связыванию сущности и значения: R3-A на новых синтетических фактах повышает macro-F1 с 0.6597 до 0.7837, R2-CE с 0.6369 до 0.7756. B7, напротив, падает с 0.5090 до 0.4360: понижение FP само по себе не означает проверку фактов. Локальные claims помогают R3-A на некоторых контролируемых длинных ответах, но не решают coreference и ошибки внутри одного длинного предложения. Natural transfer улучшений не доказан.

## Протокол и данные

План EXPERIMENT_R4_PLAN.md и models/phase2b_r4/freeze.json сохранены до основной серии. 3 неизменных inner checkpoints ×4 входных режима ×3 набора =36 завершённых основных прогонов. Threshold=0.37612172961235046, positive iff score>=threshold; локализация при .5. Seed42 фиксирует runtime и bootstrap, новых training seeds нет.

| Набор | Ответов | Sources | Negative/positive | Статус |
|---|---:|---:|---|---|
| Старый R3 diagnostic |640|12 (4 повторяющихся имени)|320/320|Ранее изученный synthetic diagnostic|
| Новый R4 |864|12 уникальных target parents|432/432|Новые fictional facts/rendering, frozen before inference|
| Natural inner-validation |648|108|442/206|Повторно используемый development набор внутри train|

У нового набора 432 same-answer supported/contradicted пары: оба числовых значения присутствуют в evidence, меняется их привязка к сущностям. Порядок evidence скрещён; 9 условий ×8 строк ×12 родителей. Четыре родителя на quantity/duration/distance; ни один parent не входил в supervised train трёх checkpoint. Новые имена и значения отделены от R3 fit. Общие шаблоны, distractor Zincford и повторяемый filler ограничивают языковое разнообразие: 864 строки не равны 864 независимым наблюдениям. Это не natural benchmark.

A: один и тот же нормализатор для evidence-only и symmetric. Консервативный scope: английские целые cardinal numbers при bolts, minutes, seconds, meters, kilometers; канонические digits/minutes/meters, seconds→minutes только при точной делимости 60. Произвольные единицы, decimals, температуры, отрицательные значения, ranges и approximations не преобразуются. Это узкий quantity normalizer, не универсальный парсер всех числовых выражений. Сущности/отношения/отрицания он не проверяет.

Соответствие offsets двунаправленное и интервальное: переписанный numeral/unit связан со всем исходным numeral/unit, а неизменённые символы сохраняют точное соответствие. Все режимы оцениваются на общей исходной tokenizer grid; probability проецируется через max-overlap. Такое отображение может расширить предсказанный span, это не выдуманное посимвольное выравнивание.

B: каждое предложение проверяется с непосредственными соседями и тем же полным question/evidence; только central tokens участвуют в объединении. Gold spans не выбирают окно. При явных anaphora весь ответ сохраняется. В natural только 62/648 ответа реально сокращены:538 anaphora fallbacks,48 single-sentence. Все 23 natural short-error случая попали в fallback; этот вариант B не проверяет улучшение на них.

## Таблицы сравнения

В R3 diagnostic положительных contradictions 272 и unsupported 48; в fresh все
432 positives — contradictions. `both_units` для quantity служит identity control:
конверсия единиц применяется к duration/distance. Повторы остаются внутри source,
а средние synthetic метрики зависят от заданного веса девяти условий.

Evidence = evidence-only; symmetric = evidence+answer; local = raw local-claim view. FPR считается среди supported/negative; contradiction recall ниже исключает unsupported R3 cases. Positive recall и все прочие метрики отдельно сохранены в JSON. Pair accuracy требует правильной классификации ОБОИХ членов пары при фиксированном пороге. Token/character/exact-span F1 — custom micro scorer, не официальный RAGTruth.

### Старая синтетическая диагностика

| Model | View | Macro-F1 | Positive F1 | FPR | Contradiction R / natural positive R | Correct pairs | Token / char / exact-span F1 |
|---|---|---:|---:|---:|---:|---|---|
|B7|raw|0.5107|0.5008|0.4688|0.5662|11/320|0.2678 / 0.2836 / 0.0123|
|B7|evidence|0.4911|0.3758|0.2562|0.3309|15/320|0.2368 / 0.2578 / 0.0037|
|B7|symmetric|0.4576|0.2686|0.1281|0.1949|19/320|0.0853 / 0.0599 / 0.0000|
|B7|local|0.5026|0.5671|0.6188|0.7426|11/320|0.1601 / 0.1659 / 0.0068|
|R2CE|raw|0.5081|0.5750|0.6188|0.7500|39/320|0.4906 / 0.4730 / 0.2403|
|R2CE|evidence|0.5572|0.5732|0.4781|0.6801|72/320|0.4253 / 0.3870 / 0.1280|
|R2CE|symmetric|0.5796|0.5723|0.4031|0.6434|90/320|0.3474 / 0.2562 / 0.0000|
|R2CE|local|0.5151|0.5992|0.6438|0.8088|49/320|0.4976 / 0.4767 / 0.2361|
|R3A|raw|0.6712|0.7370|0.5156|0.9118|122/320|0.6631 / 0.6733 / 0.5565|
|R3A|evidence|0.7772|0.8006|0.3250|0.9118|183/320|0.6853 / 0.7005 / 0.5331|
|R3A|symmetric|0.8292|0.8385|0.2250|0.9118|215/320|0.7204 / 0.7383 / 0.5243|
|R3A|local|0.6970|0.7500|0.4688|0.9081|136/320|0.6794 / 0.6918 / 0.5795|

### Новые синтетические факты

| Model | View | Macro-F1 | Positive F1 | FPR | Contradiction R / natural positive R | Correct pairs | Token / char / exact-span F1 |
|---|---|---:|---:|---:|---:|---|---|
|B7|raw|0.5090|0.4602|0.3866|0.4144|18/432|0.2080 / 0.2268 / 0.0044|
|B7|evidence|0.5061|0.4747|0.4282|0.4444|11/432|0.2568 / 0.3063 / 0.0074|
|B7|symmetric|0.4360|0.2342|0.1343|0.1505|11/432|0.0103 / 0.0091 / 0.0037|
|B7|local|0.5049|0.4837|0.4514|0.4630|9/432|0.0816 / 0.0960 / 0.0009|
|R2CE|raw|0.6369|0.6100|0.2870|0.5648|122/432|0.4132 / 0.3814 / 0.0623|
|R2CE|evidence|0.6539|0.6575|0.3565|0.6644|135/432|0.5269 / 0.5219 / 0.1675|
|R2CE|symmetric|0.7756|0.7400|0.0602|0.6227|245/432|0.5210 / 0.4569 / 0.0000|
|R2CE|local|0.6417|0.5982|0.2315|0.5255|127/432|0.4070 / 0.3776 / 0.0631|
|R3A|raw|0.6597|0.6581|0.3356|0.6551|139/432|0.4648 / 0.4555 / 0.2511|
|R3A|evidence|0.6514|0.6732|0.4097|0.7153|133/432|0.5504 / 0.5495 / 0.2919|
|R3A|symmetric|0.7837|0.7619|0.1134|0.6852|248/432|0.6112 / 0.6037 / 0.2645|
|R3A|local|0.6962|0.6733|0.2269|0.6227|171/432|0.5279 / 0.5148 / 0.3417|

### Natural inner-validation

| Model | View | Macro-F1 | Positive F1 | FPR | Contradiction R / natural positive R | Correct pairs | Token / char / exact-span F1 |
|---|---|---:|---:|---:|---:|---|---|
|B7|raw|0.7955|0.7302|0.1674|0.7816|—|0.5864 / 0.5897 / 0.0375|
|B7|evidence|0.7955|0.7302|0.1674|0.7816|—|0.5864 / 0.5897 / 0.0375|
|B7|symmetric|0.7955|0.7302|0.1674|0.7816|—|0.5864 / 0.5897 / 0.0375|
|B7|local|0.7919|0.7277|0.1787|0.7913|—|0.5876 / 0.5905 / 0.0357|
|R2CE|raw|0.7847|0.7106|0.1538|0.7330|—|0.5814 / 0.5844 / 0.0283|
|R2CE|evidence|0.7847|0.7106|0.1538|0.7330|—|0.5814 / 0.5844 / 0.0283|
|R2CE|symmetric|0.7847|0.7106|0.1538|0.7330|—|0.5814 / 0.5844 / 0.0283|
|R2CE|local|0.7807|0.7070|0.1629|0.7379|—|0.5819 / 0.5851 / 0.0275|
|R3A|raw|0.7799|0.7005|0.1425|0.7039|—|0.5652 / 0.5678 / 0.0388|
|R3A|evidence|0.7799|0.7005|0.1425|0.7039|—|0.5652 / 0.5678 / 0.0388|
|R3A|symmetric|0.7799|0.7005|0.1425|0.7039|—|0.5652 / 0.5678 / 0.0388|
|R3A|local|0.7760|0.6969|0.1516|0.7087|—|0.5643 / 0.5671 / 0.0389|

В natural нормализатор изменил **0 evidence и 0 answer**: одинаковые цифры A означают отсутствие вмешательства, а не подтверждённый natural transfer. Precision/recall/F1, accuracy, AUROC, Average Precision (не trapezoidal PR-AUC), confusion matrices и denominators доступны во всех 36 reports/phase2b_r4/*.json.

### Парные интервалы

2000 source-level draws, seed42, percentile95; одинаковые draws внутри сравнения. Интервалы условны на checkpoints, порогах и конструкции диагностики. Не отражают training-seed variability или bias многократного исследования.

| Dataset | Model | Symmetric Δ macro-F1 [95% CI] | Local Δ macro-F1 [95% CI] |
|---|---|---|---|
|old|B7|-0.0532 [-0.1228, +0.0018]|-0.0082 [-0.0189, +0.0000]|
|old|R2CE|+0.0715 [+0.0454, +0.0947]|+0.0070 [-0.0036, +0.0170]|
|old|R3A|+0.1580 [+0.1205, +0.1902]|+0.0257 [+0.0014, +0.0497]|
|fresh|B7|-0.0731 [-0.0871, -0.0614]|-0.0041 [-0.0127, +0.0040]|
|fresh|R2CE|+0.1387 [+0.0991, +0.1691]|+0.0048 [-0.0208, +0.0295]|
|fresh|R3A|+0.1240 [+0.0972, +0.1462]|+0.0365 [+0.0167, +0.0561]|
|natural|B7|+0.0000 [+0.0000, +0.0000]|-0.0036 [-0.0133, +0.0058]|
|natural|R2CE|+0.0000 [+0.0000, +0.0000]|-0.0039 [-0.0118, +0.0029]|
|natural|R3A|+0.0000 [+0.0000, +0.0000]|-0.0039 [-0.0118, +0.0029]|

Для R3-A old при более консервативных 4 name clusters CI symmetric Δ macro-F1=[0.1120, 0.1903]. Абсолютные интервалы, paired localization/pair-accuracy intervals, valid/undefined draws и symmetric−evidence localization comparisons сохранены в analysis_*.json.

## Ошибки и ограничения, которые удалось отделить

1. **Асимметрия представления.** R3-A fresh evidence_words: FPR1.0→0 при evidence normalization; answer_words остаётся FPR1.0, пока не нормализован answer. В both_words evidence-only даже портит согласованный вход: FPR0→1. Symmetric возвращает FPR0 и contradiction recall.875 во всех четырёх числовых условиях, 42/48 правильных пар в каждом. Это проверка отношений: оба конкурирующих числа остаются в документе. Однако 6/48 неправильных пар в каждом коротком canonical условии сохраняются.

2. **Нормализация не заменяет обученное связывание.** У B7 fresh symmetric FPR падает.3866→.1343 вместе с contradiction recall.4144→.1505, token F1.2080→.0103. Это провал, а не полезное снижение тревог.

3. **Локализация улучшилась не во всех смыслах.** R3-A fresh token F1.4648→.6112 (paired Δ CI[.0867,.1986]); exact-span.2511→.2645 (CI разности[-.0778,.1091]). На old token.6631→.7204, но exact-span .5565→.5243 (Δ CI[-.1185,.0537]). У R2-CE symmetric exact-span F1=0 на обоих synthetic наборах: его numeric-slot обучение и предсказание числа не обеспечивают точную границу typed measurement, включающую unit в R3/R4 gold. Улучшение response F1 не означает сохранение всех span метрик.

4. **Length/max artifacts.** R3-A long12: raw20/48 correct pairs; long36: raw0/48 при supported FPR47/48. Local даёт 26/48 в обоих условиях и FPR0, но contradiction recall только 26/48. Gold-token hit(.5) long12 растёт 14→20/48, long36 падает 22→20/48. Общий short-error response recall может падать после удаления ложного max на правильном filler; это не следует путать с потерей всех локализованных ошибок.

5. **Нерешённые случаи.** Fresh coreference36: R3-A только 2/48 contradictions detected,0/48 gold hits; full fallback сохраняет этот провал. clause36:18/48 detected,8/48 gold hits; sentence extraction не делит внутренние factual clauses, поэтому ничего не меняет. Semantic coreference и sub-sentence verification этим пакетом не решены.

### Short-error slice

Примеры R3-A fresh (все изменения, а не только удачные, сохранены в error JSONL):

| Case | Gold | View | Raw → new risk | Результат |
|---|---:|---|---|---|
| `Alderwick:answer_words:o0:a0:e0` — supported 112 bolts словами |0|symmetric|0.916409 → 0.047780|FP исправлен|
| `Alderwick:evidence_words:o1:a1:e0` — ответ 118 при target 112 |1|symmetric|0.886420 → 0.349977|Новый FN, неверная привязка не распознана|
| `Alderwick:long36:o0:a1:e0` — та же неверная величина в длинном ответе |1|local|0.511229 → 0.289975|Новый response FN|

Всего R3-A fresh: symmetric исправляет 122 решения и создаёт 13 новых ошибок;
evidence-only — 80 исправлений и 86 новых; local — 55 и 22. На natural local
исправляет 1 решение и создаёт 4 новые ошибки. Это не универсальное улучшение.

Positive answer>=500 characters, union of annotated characters<=5%; response TP при.3761217 и gold-token hit при.5 — разные показатели. Ни один результат не использует gold для выбора inference окна.

| Dataset | Model | N | Raw response TP / gold hit | Local response TP / gold hit |
|---|---|---:|---|---|
|old|B7|48|0 / 0|48 / 1|
|old|R2CE|48|0 / 0|16 / 9|
|old|R3A|48|30 / 8|29 / 22|
|fresh|B7|144|33 / 4|43 / 0|
|fresh|R2CE|144|47 / 20|32 / 21|
|fresh|R3A|144|66 / 30|46 / 28|
|natural|B7|23|17 / 11|17 / 11|
|natural|R2CE|23|14 / 8|14 / 8|
|natural|R3A|23|15 / 9|15 / 9|

Данные по token-length bins<=64/65–128/>128 находятся в analysis_*.json; sentence-count bins и eligible/fallback subsets — eligibility_and_projection.json. Полные исправленные и новые ошибки с answer/evidence/labels/scores/offsets сохранены в ignored predictions/phase2b_r4/errors_*.jsonl.

## Стоимость и воспроизводимость

RTX3060 12GB, BF16 autocast/FP32 weights, batch2; checkpoints не менялись. У local runtime в JSON измеряет дополнительный проход, поскольку fallback использует уже вычисленный raw. В следующей таблице local time включает raw+incremental local, чтобы не выдавать cache reuse за бесплатный inference. Это измеренный последовательный двухпроходный путь, не оптимизированный single-pass dispatcher. Model loading, shared original-token-grid encoding, hashing и disk writes вне timing; latency — amortized, не online SLA.

| Dataset | Model | raw/evidence/symmetric/local seconds | local responses/s | Peak allocated VRAM GB |
|---|---|---|---:|---:|
|old|B7|14.71 / 16.17 / 14.61 / 56.07|11.41|0.892|
|old|R2CE|14.34 / 14.51 / 14.72 / 54.47|11.75|0.892|
|old|R3A|14.74 / 15.35 / 15.48 / 57.85|11.06|0.892|
|fresh|B7|19.91 / 20.60 / 20.48 / 129.14|6.69|0.901|
|fresh|R2CE|19.82 / 20.99 / 22.26 / 127.84|6.76|0.901|
|fresh|R3A|21.05 / 21.08 / 20.90 / 129.95|6.65|0.901|
|natural|B7|16.65 / 16.33 / 17.61 / 28.61|22.65|0.953|
|natural|R2CE|17.69 / 19.43 / 18.16 / 29.75|21.78|0.953|
|natural|R3A|16.39 / 15.72 / 16.85 / 28.23|22.95|0.953|

Сумма измеренных 36 stages: 963.16s (16.05min), peak allocated 0.953GB. Это не суммарный wall time всей работы. Нет OOM, discarded evidence или потерянных answer characters; native metadata фиксируют 0 truncated tokens. Local fresh создаёт 4800 views; natural473, с явным fallback. Peak RSS/reserved VRAM, batch latency p50/p95 и view throughput — в runtime JSON.

**Исторический B7 воспроизведён отдельно.** При исходном batch1 все token probabilities совпали с сохранёнными R3 побитово: macro-F1=.7959346775, CM=[[367,75],[44,162]]. Preregistered batch2 R4 даёт.7954887218, CM=[[368,74],[45,161]]: поменялись 2 решения около порога. Максимальное расхождение score между batch sizes=.091526. Это измеренная sensitivity BF16/padding batch shape; seed и weights неизменны. Primary R4 batch2 и порог после аудита не заменялись более выгодным вариантом. См. B7_batch_audit.json и models/phase2b_r4/BATCH_AUDIT.md.

### Исправления реализации и сохранение артефактов

Первый B7 pilot остановлен после трёх old-input stages: blanket `to` guard блокировал `According to`. Ошибка обнаружена до просмотра метрик; исходные freeze/code/predictions/logs сохранены в parser_pilot и source ZIP. Основная серия выполнена после предзарегистрированной коррекции. Дополнительный adversarial parser audit до просмотра метрик выявил mixed word/digit ranges. После завершения frozen inference guard усилен; для всех 4304 evidence/answer полей трёх наборов подтверждено точное равенство текста, offset maps и edit records между executed/hardened кодом. Предсказания не пересчитаны и не переименованы как новые результаты. Обе версии кода и proof сохранены. Helper queue.py был переименован в serial_inference.py после stdlib-name collision; exclusive log creation предотвратила дублирование GPU subprocess.

Старый freeze остаётся неизменным и отвергается изменённым source hash. Новый --output-name создаёт отдельный reproduction namespace; это не обход старого guard. Проверены 171 старый R3 artifact,3 checkpoint и 36 prediction hashes; полный executed source snapshot восстановим. Source-only архив research_snapshots/R4_source_20261024.zip содержит код, планы, документацию и helpers, без datasets/weights/predictions.

## Проверки

- Research environment + реальные dataset/tokenizer opt-ins: **178 passed** (71.18s).
- Offline environment: **172 passed,6 skipped** (56.54s), без downloads/GPU requirement.
- Новые R4 regression tests:35; ranges/approximation, exact conversions, двунаправленные offsets, projection, grouped pairs, central coverage/coreference.
- Ruff: passed. pip check в обоих окружениях: no broken requirements.
- CI .github/workflows/ci.yml существует, tracked, Python3.11/3.12, offline; remote CI не запускался и успешный remote run не заявляется.

## Один следующий training experiment — предложение, не запуск

**Гипотеза:** supervised CE на raw+canonical views научит relation-trained encoder сохранять числовую поддержку без необходимости асимметричного совпадения token forms. Positive-only consistency MSE из R3 не повторять: он уже ухудшил разделение. Здесь обе truth labels получают явные targets.

**Контроль:** одинаковый старт B7 inner42 и рецепт R3-A, matched source/batch order/optimizer/updates/seed42. Control предъявляет raw+raw повтор; treatment raw+symmetrically canonicalized view. Оба получают те же natural inner-fit 2568 и R3 authored-fit parents, без новых heldout targets. Все производные примеры остаются с parent. Loss — тот же weighted token CE, фиксированный positive weight3.370202392225359; question/evidence/special masked. Нормализованные targets получаются через проверенные interval maps; не делать entailed labels из отсутствия hallucination spans.

**Оценка и gate:** до fits зафиксировать новые train-side parents/формулировки и небольшой вручную проверенный natural quantity audit, где преобразование реально действует. R4 использовать как явно reused regression, не независимое доказательство. На новых word/unit pairs: raw-input pair accuracy +>=.10, supported FPR −>=.15 при contradiction recall не хуже−.03; natural macro-F1, token и exact-span F1 не хуже−.01. Публиковать все метрики, даже если gate не пройден. Raw/canonical inference crossed для обоих arms отделит обучение от deterministic preprocessing. Этот опыт сам по себе не обещает решить coreference или short clauses; расширение контекста и новые losses не смешивать.

**Бюджет:** два matched one-epoch fits seed42, BF16, microbatch2/accum4, gradient checkpointing; сначала smoke/profile новых token targets. По измеренным R3 ~332s/fit и 3.17GB training peak планировать 7–10min/arm (больше effective views), hard cap20 GPU min плюс smoke. Это оценка, не измерение ещё не запущенного обучения. Seeds43/44 только отдельным следующим пакетом после gate, показывать все seeds. В R4 ничего из этого не обучалось.

## Команды воспроизведения

Ниже новый output namespace: существующие artifacts не перезаписываются. Используйте сохранённое .venv-research и локальные pinned checkpoints; модели не скачиваются. Для побитового воспроизведения именно executed версии восстановите source ZIP, указанный в final_integrity.json, в отдельный checkout; не заменяйте source hashes старого freeze.

```powershell
.venv-research\Scripts\python.exe scripts/research_r4.py freeze --output-name phase2b_r4_repro
.venv-research\Scripts\python.exe scripts/research_r4.py B7 --output-name phase2b_r4_repro
.venv-research\Scripts\python.exe scripts/research_r4.py R2CE --output-name phase2b_r4_repro
.venv-research\Scripts\python.exe scripts/research_r4.py R3A --output-name phase2b_r4_repro
.venv-research\Scripts\python.exe models/phase2b_r4/analyze.py --output-name phase2b_r4_repro
.venv-research\Scripts\python.exe models/phase2b_r4/supplement.py --output-name phase2b_r4_repro
$env:GROUNDCHECK_RAGTRUTH_DIR="data/raw/ragtruth"
$env:GROUNDCHECK_RESEARCH_MODELS="1"
.venv-research\Scripts\python.exe -m pytest -q -o cache_dir=models/r4_check_cache --basetemp=models/r4_check_tmp
.venv-research\Scripts
uff.exe check .
.venv-research\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pip check
```

Bootstrap/error analysis implementation: models/phase2b_r4/analyze.py; eligible/length audit: supplement.py; integrity: verify_final.py. Эти helpers сохранены также в source-only архиве; существующие outputs защищены от overwrite. R4 завершён как исследовательский пакет, проект не объявлен завершённым.

## Изменённые файлы

Добавлены: EXPERIMENT_R4_PLAN.md; scripts/research_r4.py; src/grounded_hallucination_lab/research/numeric_local.py; src/grounded_hallucination_lab/research/numeric_local_suite.py; tests/test_numeric_local.py; docs/PHASE2B_R4_RESULTS.md. Обновлены docs/EVALUATION_PROTOCOL.md, docs/PROGRESS.md, docs/DECISIONS.md, RESEARCH_NOTES.md. Локальные модели не менялись; новые reports/predictions и служебные manifests находятся в phase2b_r4 directories. Ничего не закоммичено и не опубликовано.
