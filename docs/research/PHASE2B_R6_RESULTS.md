# R6 — Entity–Property–Value Binding

Дата: 2026-10-24. Пакет выполнен, исследование продолжается. План:
`EXPERIMENT_R6_PLAN.md`; freeze: `models/phase2b_r6/freeze.json`.

**Вывод:** разнообразие evidence renderers существенно улучшило перенос внутри
нового synthetic relation benchmark: R6-B macro-F1=.863807 против R6-A=.598482,
Δ+.265325, paired95%[+.2076,+.3199]. Однако критерий общего успеха НЕ выполнен:
natural macro-F1=.784163 против .797171 у A (Δ−.013008,95%[−.03135,+.00655]),
а short-error recall и gold-hit на длинных synthetic ответах ухудшились.
Gate false; seeds43/44 не запускались. Это сильный локальный результат по
binding, а не доказательство улучшения универсального detector или SOTA.


## Numerical integrity: измерено до обучения

Один неизменный B7 inner42 checkpoint, 50 одинаковых входов: 32 inner-fit,
16 прежних R5 synthetic и два явно исторических численных sentinel из inner-val.
Это намеренно стрессовый аудит, не оценка распространённости нестабильности.
12 прогонов: BF16/FP32 × batch1/batch2/fixed-padding1024 × два повтора.
TF32 выключен; исходные predictions и checkpoints не менялись.

| Сравнение | Max response score delta | Max token probability delta | Threshold flips / 50 |
|---|---:|---:|---:|
| BF16 batch1→2 | .02430737 | .20602372 | 2 |
| BF16 batch1→padding1024 | .01531887 | .20602372 | 3 |
| FP32 batch1→2 | .000002682 | .000018060 | 0 |
| FP32 batch1→padding1024 | .000001371 | .000005245 | 0 |
| BF16→FP32 batch1 | .02201822 | .14644387 | 2 |
| BF16→FP32 batch2 | .02099216 | .06394172 | 2 |

Все шесть одинаковых повторов совпали точно. BF16 детерминирован при неизменной
форме входа, но чувствителен к batching/padding; на max-token risk это способно
менять решения у порога. FP32 прошёл заранее заданный gate: max token delta<1e-4,
нулевые flips и точный повтор. Это проверка данного GPU/стека/набора, не гарантия
побитового совпадения на любом железе. Лог: `models/phase2b_r6/audit/result.json`.

Новый общий протокол: **FP32, batch1, no artificial padding, TF32off**. Все четыре
модели оцениваются заново. Порог ответа прежний: score>=.37612172961235046;
локализация p>=.5. Никакого подбора порога по transfer или natural. Исторические
BF16 метрики остаются самостоятельными историческими результатами.

## Данные, контроль и границы вывода

Natural replay2568/428 sources + synthetic768/16 parents =3336 записей в каждом
arm,1173 positives. Две одинаковые exposures на запись,3336 microbatches,
834 updates. Инициализация обоих — B7 inner42. В A старый declarative renderer,
в B шесть evidence renderers. Worlds, ответы, questions, labels и gold spans
совпадают; swaps присутствуют в обоих arms. Разнообразие answer syntax оставлено
для переноса, чтобы не менять две компоненты сразу. Нет canonicalization,
ranking/consistency loss, нового retrieval или aggregation.

Natural inner648/108,206 positives; sensitivity642 без известного shared-passage
source15181. Transfer768/16 новых fictional parents,384 positives,384 пары;
шесть новых evidence families, новый answer renderer, новые значения и имена.
Категории: entity/property swaps, shared values, similar names, prose/order,
long correct answer + short wrong claim. Оба значения есть в evidence.

Каждое утверждение сверено с explicit fact ledger. Exact gold span — ошибочная
числовая фраза с единицей. Answer идентичен в supported/contradicted мирах;
каждый evidence встречается с обоими answer values. Все marginal lookup upper
bounds равны .5: answer/evidence text, parent/name, value, order, template,
char length и claim position. Отдельно проверены совпадение answer с первым/последним
числом evidence, greater-value indicator и позиция target fact: lookup=.5.
Это исключает эти одиночные shortcuts, но не все
возможные joint shortcuts. Script: `models/phase2b_r6/audit_controls.py`.

Answer targets совпадают побитово:443201 tokens,34212 positive targets на один
проход; SHA в `control_audit.json`. Остался confound: input tokens1390069→1395589
(+5520,+0.397%); меняются evidence length, позиции и padding, поэтому чистый
эффект синтаксиса отдельно от длины не идентифицирован. Все training inputs
один window, coverage100%, max1008 tokens. Profile обоих arms прошёл, peak
allocated~3.15GB. Training BF16 matched, inference FP32 для всех.

Синтетика не является новыми natural documents. Шестнадцать parents — только
16 независимых bootstrap units; многочисленные варианты не увеличивают это число.
Long ответы повторяют три корректные filler-фразы; свойства и units знакомы по
train. Это ограничивает перенос вывода на естественную прозу. Категории частично
перекрываются. Natural inner-validation уже использовалась в предыдущих пакетах.
Новые R6 модели не оцениваются и не выбираются по outer calibration/dev/test.
Отдельное исключение в старых автоматических регрессиях описано ниже.

## Метрики и проверка воспроизводимости

Основные разности: paired source bootstrap2000, seed42, percentile95; абсолютные
интервалы и разности относительно R6-A/B7. Macro-F1 не подменяет positive F1.
Pair accuracy = оба решения верны при фиксированном пороге. Claim-local и slot-local
separation — max probability в известном диапазоне contradiction минус support;
диапазоны используются исключительно в анализе, не на входе detector.
Token/character/exact-span scorer — существующий custom scorer, не официальный.
Short natural: >=500 chars и gold nonspace<=5% (20 positives). Gold-hit требует
token p>=.5 с пересечением gold character. Synthetic long slice анализируется
целиком, включая более короткие варианты с12 filler sentences.

Offline pytest: **183 passed,7 skipped (58.20s)**. Research pytest с локальными
model integrations: **190 passed (67.10s)**. Ruff и pip check обеих сред прошли.
Отдельно проверено реальным вызовом CLI: seed43 отвергнут `repeat-seed gate closed`
до model loading и создания checkpoint directory. Analysis smoke с известными
идеальными предсказаниями проверил pair/claim metrics и нулевые paired differences.

Прозрачность scope тестов: включённый `GROUNDCHECK_RAGTRUTH_DIR` активировал
существующий `test_real_snapshot_metrics_and_membership`, который пересчитывает
**исторический majority baseline на outer dev** и сравнивает его с Phase1 fixture.
Это не R6 inference и не новые архитектурные результаты; он выполнен после всех
зафиксированных R6 predictions и не влиял на выбор. Тем не менее это обращение
к outer dev внутри старой регрессии, поэтому нельзя буквально утверждать, что
никакой тест dev не читал. Official test labels остались sealed. В будущих строго
train-side research проверках не включать эту opt-in переменную; model integration
flag независим. Саму старую регрессию не удаляли и не ослабляли.

CI `.github/workflows/ci.yml` существует и tracked, Python3.11/3.12, offline pytest,
Ruff, pip check. Новый test_binding.py попадает в общий pytest. Remote CI не проверялся.


## Фактические результаты и интервалы

Один matched training seed42; все сравнения ниже используют одинаковый FP32
inference и фиксированный прежний threshold. Natural n648 (442/206), transfer
n768 (384/384). FNR=1−positive recall, также явно сохранён в JSON. AP — Average
Precision, не трапецеидальная PR-AUC. Полные absolute/paired CIs всех метрик:
`reports/phase2b_r6/analysis_{natural,transfer}.json`.

### natural

| Method | Accuracy | Macro-F1 | Positive P / R / F1 | FPR | AUROC / AP | CM |
|---|---:|---:|---:|---:|---:|---|
| B7 | 0.8164 | 0.7959 | 0.6835 / 0.7864 / 0.7314 | 0.1697 | 0.8765 / 0.7903 | [[367, 75], [44, 162]] |
| R5A | 0.8256 | 0.7954 | 0.7409 / 0.6942 / 0.7168 | 0.1131 | 0.8752 / 0.7927 | [[392, 50], [63, 143]] |
| R6As42 | 0.8241 | 0.7972 | 0.7233 / 0.7233 / 0.7233 | 0.1290 | 0.8734 / 0.7826 | [[385, 57], [57, 149]] |
| R6Bs42 | 0.8102 | 0.7842 | 0.6912 / 0.7282 / 0.7092 | 0.1516 | 0.8735 / 0.7834 | [[375, 67], [56, 150]] |

| Method | Token / character / exact-span F1 | Pair accuracy | Response / claim / slot separation | Short TP / gold-hit / n |
|---|---:|---:|---:|---:|
| B7 | 0.5868 / 0.5906 / 0.0373 | NA | NA / NA / NA | 15 / 9 / 20 |
| R5A | 0.5734 / 0.5754 / 0.0446 | NA | NA / NA / NA | 12 / 7 / 20 |
| R6As42 | 0.5768 / 0.5804 / 0.0376 | NA | NA / NA / NA | 15 / 8 / 20 |
| R6Bs42 | 0.5800 / 0.5822 / 0.0546 | NA | NA / NA / NA | 14 / 6 / 20 |

| Method | Macro-F1 absolute 95% CI | Macro-F1 Δ vs A 95% CI | Macro-F1 Δ vs B7 95% CI |
|---|---:|---:|---:|
| B7 | [0.7656, 0.8280] | [-0.0241, 0.0218] | reference |
| R5A | [0.7580, 0.8333] | [-0.0214, 0.0175] | [-0.0267, 0.0239] |
| R6As42 | [0.7619, 0.8314] | reference | [-0.0218, 0.0241] |
| R6Bs42 | [0.7481, 0.8192] | [-0.0313, 0.0065] | [-0.0302, 0.0060] |

R6-B − R6-A paired95%: token_f1 [-0.0047, 0.0126]; character_f1 [-0.0068, 0.0119]; exact_span_f1 [0.0019, 0.0335]; pair_accuracy [NA, NA]; mean_paired_separation [NA, NA]; positive_separation_fraction [NA, NA]; short_gold_hit [-0.2727, 0.0000]; short_response_recall [-0.1765, 0.0000]; claim_separation [NA, NA]; slot_separation [NA, NA]

Error transitions B vs A: {'fixed': 8, 'new_error': 17}

### transfer

| Method | Accuracy | Macro-F1 | Positive P / R / F1 | FPR | AUROC / AP | CM |
|---|---:|---:|---:|---:|---:|---|
| B7 | 0.4974 | 0.4399 | 0.4928 / 0.1771 / 0.2605 | 0.1823 | 0.5213 / 0.5151 | [[314, 70], [316, 68]] |
| R5A | 0.5052 | 0.4342 | 0.5179 / 0.1510 / 0.2339 | 0.1406 | 0.6014 / 0.5614 | [[330, 54], [326, 58]] |
| R6As42 | 0.6367 | 0.5985 | 0.5845 / 0.9453 / 0.7224 | 0.6719 | 0.8454 / 0.8568 | [[126, 258], [21, 363]] |
| R6Bs42 | 0.8646 | 0.8638 | 0.9294 / 0.7891 / 0.8535 | 0.0599 | 0.9388 / 0.9475 | [[361, 23], [81, 303]] |

| Method | Token / character / exact-span F1 | Pair accuracy | Response / claim / slot separation | Short TP / gold-hit / n |
|---|---:|---:|---:|---:|
| B7 | 0.0024 / 0.0015 / 0.0000 | 0.0182 | 0.0067 / 0.0074 / 0.0081 | 50 / 0 / 64 |
| R5A | 0.0006 / 0.0007 / 0.0000 | 0.0104 | 0.0282 / 0.0311 / 0.0312 | 54 / 1 / 64 |
| R6As42 | 0.7468 / 0.7489 / 0.5481 | 0.2812 | 0.3251 / 0.3264 / 0.3264 | 43 / 38 / 64 |
| R6Bs42 | 0.7869 / 0.7973 / 0.6783 | 0.7344 | 0.6518 / 0.6518 / 0.6519 | 36 / 29 / 64 |

| Method | Macro-F1 absolute 95% CI | Macro-F1 Δ vs A 95% CI | Macro-F1 Δ vs B7 95% CI |
|---|---:|---:|---:|
| B7 | [0.4235, 0.4538] | [-0.1973, -0.1215] | reference |
| R5A | [0.4229, 0.4469] | [-0.2072, -0.1231] | [-0.0237, 0.0124] |
| R6As42 | [0.5626, 0.6381] | reference | [0.1215, 0.1973] |
| R6Bs42 | [0.8280, 0.8955] | [0.2076, 0.3199] | [0.3829, 0.4636] |

R6-B − R6-A paired95%: token_f1 [-0.0307, 0.1046]; character_f1 [-0.0172, 0.1050]; exact_span_f1 [0.0655, 0.1872]; pair_accuracy [0.3620, 0.5365]; mean_paired_separation [0.2423, 0.4050]; positive_separation_fraction [0.0104, 0.0756]; short_gold_hit [-0.3438, 0.0469]; short_response_recall [-0.3125, 0.0781]; claim_separation [0.2399, 0.4042]; slot_separation [0.2399, 0.4041]

Error transitions B vs A: {'fixed': 243, 'new_error': 68}

| Category | B7 macro | R5-A macro | R6-A macro | R6-B macro | A→B FPR | A→B recall |
|---|---:|---:|---:|---:|---:|---:|
| entity_swap | 0.3792 | 0.3505 | 0.6213 | 0.9375 | 0.6719→0.0469 | 1.0000→0.9219 |
| long_short | 0.4571 | 0.4330 | 0.6403 | 0.7190 | 0.3906→0.1094 | 0.6719→0.5625 |
| property_swap | 0.3505 | 0.3333 | 0.5873 | 0.9062 | 0.7188→0.0781 | 1.0000→0.8906 |
| prose_order | 0.3750 | 0.3671 | 0.5515 | 0.9766 | 0.7656→0.0156 | 1.0000→0.9688 |
| shared_value | 0.3392 | 0.3333 | 0.6537 | 0.6929 | 0.6250→0.0469 | 1.0000→0.4688 |
| similar_name | 0.4113 | 0.3505 | 0.4730 | 0.9297 | 0.8594→0.0625 | 1.0000→0.9219 |

### Runtime (model scoring only; excludes loading, encoding, disk and bootstrap)

| Dataset / method | Seconds | Responses/s | p50 / p95 batch latency ms | Peak allocated / reserved GiB |
|---|---:|---:|---:|---:|
| natural / B7 | 25.43 | 25.49 | 35.58 / 52.10 | 0.644 / 0.672 |
| natural / R5A | 25.16 | 25.76 | 35.14 / 51.45 | 0.644 / 0.672 |
| natural / R6As42 | 25.46 | 25.46 | 35.87 / 51.83 | 0.644 / 0.672 |
| natural / R6Bs42 | 24.93 | 26.00 | 34.98 / 51.65 | 0.644 / 0.672 |
| transfer / B7 | 26.66 | 28.81 | 32.76 / 42.78 | 0.622 / 0.674 |
| transfer / R5A | 26.16 | 29.36 | 32.46 / 40.21 | 0.622 / 0.674 |
| transfer / R6As42 | 26.41 | 29.08 | 32.49 / 41.72 | 0.622 / 0.674 |
| transfer / R6Bs42 | 26.45 | 29.03 | 32.66 / 41.40 | 0.622 / 0.674 |

## Что улучшилось и что не сохранилось

1. R6-A против B7 показывает вклад нового supervised relation curriculum:
   transfer MF1+.15855,95%[+.1215,+.1973], но supported FPR67.19% — модель часто
   тревожится почти на любые новые relation statements. Это не чистая абляция
   относительно R5-A: набор synthetic фактов и бюджет отличаются от R5.
2. Единственная matched ablation A→B — evidence rendering. Pair accuracy108/384→282/384;
   Δ+.453125,95%[+.3620,+.5365]. Claim-local separation .326367→.651841,
   Δ+.325474,95%[+.2399,+.4042]. AUROC .8454→.9388, ΔCI[+.0524,+.1367];
   AP .8568→.9475, ΔCI[+.0496,+.1348]. Поэтому выигрыш не сводится к подавлению
   всех positive predictions: ranking и counterfactual separation тоже улучшились.
3. Цена на transfer: FPR258/384→23/384, но contradiction recall363/384→303/384.
   Исправлено243 ошибки (237 FP,6 FN), появилось68 (2 FP,66 FN). Общая token F1
   выросла .7468→.7869, но её CI разности включает0. Exact-span F1 .5481→.6783,
   CI разности[+.0655,+.1872]. При этом long-short gold-hit38/64→29/64,
   response TP43/64→36/64. Aggregate localization скрывает эту деградацию.
4. На natural FP57→67,TP149→150. FPR increase+.02262,95%[+.00674,+.04054].
   AUROC .87338→.87350 и AP .78258→.78342 почти неизменны. Это совместимо со
   сдвигом operating point/score distribution, но не доказывает причинный механизм.
   Macro CI включает0: статистически доказанный общий вред не заявляем, однако
   заранее заданная retention tolerance не выполнена. Natural short TP15→14,
   gold-hit8→6 из20; B7=15/9. Exact-span aggregate вырос .0376→.0546.
5. Sensitivity642 без source15181: B7=.795557,R5-A=.794947,A=.796755,B=.783628.
   Направление вывода сохраняется. Эта sensitivity не новый отбор модели.

### Главные оставшиеся ошибки

**Повторяющиеся значения.** В shared_value B пропускает34/64 contradictions:
recall=.46875 против .8906–.9688 в других коротких категориях. Дополнительный
объект имеет те же значения, но относится к другой entity/property. Само наличие
правильного количества где-то в документе всё ещё может давать ложную поддержку.
Вместе shared_value и long_short составляют62/81 transfer FN у B. Это наблюдение
по заранее заданным категориям, не доказательство единственного causal shortcut.

**Короткая ошибка в длинном ответе.** B7:50 response TP, но0 gold-hit;49/50 FP
имеют max вне claim. R5-A:54 TP,1 gold-hit;54/54 FP вне claim. У R6-B max уже
внутри claim на всех64 positive long примерах; FP7 тоже внутри claim, однако
только36 TP и29 gold-hit. Следовательно, шум от корректного filler сильно
уменьшился, а низкая уверенность/ошибочная поддержка локального числового факта
осталась. Дальнейшая смена одного max aggregation не гарантирует исправления.

Длина сама по себе здесь не установлена как причина: n48 long примеров в bins65–128
answer tokens имеют B macro-F1=.5474, n80 более длинных — .7995. Parents/families
и позиции различаются между bins; это descriptive slices, не causal length ablation.
По family/order/claim_position/generator/annotation type есть полные slices в JSON.

Примеры (из всех сохранённых transitions, без исключений в detector):

- `r6:transfer:Bellflower:long_short:o0:a0:w1`: правильные417 bolts для evening
  shipment;425 относятся к morning/другому объекту. FP A=.6526 исправлен B=.2671.
- `r6:transfer:Bellflower:long_short:o1:a1:w1`: в том же мире неправильные425 bolts
  внутри длинного ответа, gold[819,828). A=.4969 обнаруживал, B=.2827 пропускает.
- `r6:transfer:Dahlia:long_short:o1:a0:w1`: для evening shipment Dawlish нужно453,
  а445 относится к morning shipment и другому объекту. A=.8926→B=.0112: новый FN,
  хотя оба значения доступны в evidence. Gold[256,265).
- Natural12026: отмеченное аннотаторами дополнительное утверждение, FN исправлен
  (.2957→.5206). Natural12627: ответ с несколькими ценовыми ставками без gold spans,
  новый FP (.3410→.4642). Это ошибки относительно RAGTruth policy include/all;
  отсутствие gold spans не объявляется доказательством entailment каждой фразы.

Полные answer/evidence/spans/scores всех изменившихся решений — ignored
`predictions/phase2b_r6/errors_B_minus_A_{natural,transfer}.jsonl`.
`max_token_audit.json` содержит counts по каждой категории.

### Gate и следующая гипотеза

Gate false по пяти условиям: natural point retention против A, natural lower CI,
natural retention против B7, long-short response recall и long-short gold-hit.
Остальные11 условий прошли. Никаких rescue thresholds или seeds43/44.
Bootstrap учитывает16/108 source groups и2000 общих draws, но не training-seed
variability, предыдущий model selection или ограниченность authored templates.

**Следующая проверяемая гипотеза:** ложная поддержка возникает, когда ошибочное
значение совпадает со значением у нерелевантной entity/property; число повторов
усиливает этот эффект независимо от истинного target relation. Следующий пакет
сначала должен сделать *inference-only causal ablation* на неизменном R6-B:
для одной и той же проверяемой фразы/target fact сравнить равноразмерные contexts,
где добавленные нерелевантные факты содержат либо competing value, либо другое
3-digit value. Число entities/facts, позиции, шаблон и answer length совпадают;
truth/spans неизменны. Перекрестить supported/contradicted и короткий/длинный answer,
заморозив новые parents до inference. Основной endpoint — падение claim-local risk
на contradictions и paired label flips; это проверяет механизм, а не новый threshold.

Если causal эффект подтвердится, один следующий matched CE experiment:
фиксированный R6-B curriculum + natural replay, raw/raw контроль против
raw/label-preserving distractor-intervention views; только train, одинаковый B7
init,seed42,updates и class weights, без новых ranking/consistency losses.
Ресурс: сначала profile, затем ~25 GPU-min на пару при текущем объёме; не масштабировать.
Критерий — gain на новых distractor families с сохранением natural и long-short
retention, с тем же строгим gate. Если inference-аудит не подтвердит механизм,
это обучение не запускать. Ни аудит, ни следующий fit в R6 не выполнены.

## Артефакты и сохранность

Training A610.4349s,B606.5164s (~20.28GPU-min суммарно),3336 microbatches/834updates.
Peak allocated3167915520 bytes (~2.950GiB),reserved3374317568 bytes (~3.143GiB).
Peak process RSS A2.715GB/B2.744GB. Данные и модели локальные, платных ресурсов нет.
Конфигурации после выполнения проверены: отличаются только `arm`; порядок,
answer targets, replay, optimizer и budgets совпадают. Модели не были перезапущены.

- A checkpoint `models/phase2b_r6/R6As42/best`, SHA256
  `2755efe431e464ccc8cc9b48a655c711af6fb79932886952e803fbb2ef71c677`.
- B checkpoint `models/phase2b_r6/R6Bs42/best`, SHA256
  `86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e`.
- Configs, batch logs, resource/profile reports, numerical audit predictions,
  frozen corpus hashes, control audits и queue events — `models/phase2b_r6/`.
- Predictions — `predictions/phase2b_r6/`; metrics/CIs — `reports/phase2b_r6/`.
- Итоговый source archive `research_snapshots/R6_source_20261024.zip` и JSON inventory
  сохраняют также ignored experiment helpers; datasets/weights/cache туда не входят.
  `models/phase2b_r6/R6_artifact_manifest.json` — проверяемый inventory результатов.
  Старые125 artifacts R5 перепроверяются, старые source guards не обходятся.

Изменённые/новые рабочие файлы: `research/binding.py`, `research/stable_inference.py`,
`scripts/research_r6.py`, `tests/test_binding.py`, `EXPERIMENT_R6_PLAN.md`, этот отчёт,
`docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/EVALUATION_PROTOCOL.md`, `RESEARCH_NOTES.md`.
Helpers в models сохранены в source archive; сами weights/data/predictions остаются
ignored. Ничего не коммитилось, не пушилось и не публиковалось.

## Воспроизведение

Все команды из корня репозитория, `.venv-research` из существующего lock.
Выходные пути immutable: не запускать повторно поверх сохранённых artifacts.
Для полного replay — отдельная копия workspace с исходными локальными весами и
данными, без R6 outputs; baseline hashes проверяются автоматически.

```powershell
.venv-research\Scripts\python.exe models/phase2b_r6/inference_audit.py
.venv-research\Scripts\python.exe scripts/research_r6.py freeze
.venv-research\Scripts\python.exe models/phase2b_r6/run_packet.py
.venv-research\Scripts\python.exe models/phase2b_r6/audit_controls.py
.venv-research\Scripts\python.exe models/phase2b_r6/audit_value_order.py
.venv-research\Scripts\python.exe models/phase2b_r6/analyze.py
.venv-research\Scripts\python.exe models/phase2b_r6/gate.py
.venv-research\Scripts\python.exe models/phase2b_r6/error_audit.py
.venv-research\Scripts\python.exe models/phase2b_r6/make_tables.py
.venv-research\Scripts\python.exe models/phase2b_r6/check_repeat_guard.py
.venv-research\Scripts\ruff.exe check .
.venv-research\Scripts\python.exe -m pip check
```

GPU очередь: profile A/B → train A42/B42 → evaluate B7/R5-A/R6-A/R6-B.
Отдельная команда обучения: `scripts/research_r6.py train --arm A --seed 42`;
inference: `scripts/research_r6.py evaluate --method R6A --seed 42`.
Не повторять завершённые fits. Mid-epoch optimizer resume не реализован.

Offline test command (no outer data):
```powershell
.venv\Scripts\python.exe -m pytest -q -o cache_dir=models/phase2b_r6/offline_cache --basetemp=models/phase2b_r6/offline_tmp
```
Actually executed opt-in test command (includes the legacy dev majority regression
noted above; do not repeat when strict no-outer access is required):
```powershell
$env:GROUNDCHECK_RAGTRUTH_DIR='data/raw/ragtruth'
$env:GROUNDCHECK_RESEARCH_MODELS='1'
.venv-research\Scripts\python.exe -m pytest -q -o cache_dir=models/phase2b_r6/research_cache --basetemp=models/phase2b_r6/research_tmp
```
For future model-only integrations, omit/unset GROUNDCHECK_RAGTRUTH_DIR.
