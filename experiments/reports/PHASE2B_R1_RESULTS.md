# Phase 2B+ R1 — промежуточный научный отчёт

Runs:2026-10-23; recovery и завершение пакета:2026-10-24. Статус: R1 выполнен, исследование проекта продолжается. Это исследовательский checkpoint, не release и не завершение проекта.

## Основной результат

Контекст действительно помогает: matched inner ModernBERT B7 превосходит answer-only B8 на 0.089812 macro-F1; paired source 95% CI [0.045754, 0.135162]. Но B8 сохраняет macro-F1 0.7061, а length-only prior — 0.7172. Поэтому хорошая response-level метрика сама по себе не доказывает точную проверку фактов. B8 удаляет одновременно question и evidence: разность не является причинным процентом вклада одного evidence.

На контролях с неизменным ответом и изменённым отношением в evidence B7 гораздо слабее MiniCheck. Выбранная по этому наблюдению контрфактическая донастройка C1 идеально выучила синтетические training-примеры, но не улучшила natural inner-validation относительно matched continuation C0. Увеличивать число эпох или объявлять эту модель улучшением оснований нет.

## Протокол и границы вывода

Предрегистрация: EXPERIMENT_PLAN_RESUME.md; узкая гипотеза и контроль C0/C1 — EXPERIMENT_R1_CF.md. После результатов C1 отдельный read-only diagnostic зафиксирован в EXPERIMENT_R1_DIAGNOSTIC.md до новых predictions. Старый EXPERIMENT_PLAN.md и Phase2A/Phase2B артефакты сохранены. Никакой новой calibration/dev/test оценки в R1 нет.

QA, include/all, positive=hallucination. Outer: train3216/536 sources (2221/995 negative/positive), calibration810/135 (548/262), dev1008/168 (701/307), official test900/150 с закрытыми labels. Inner-fit2568/428 (1779/789), inner-validation648/108 (442/206). Все производные CF-примеры наследуют source group. Числа ниже относятся к одному и тому же648-row inner-validation, уже используемому для selection. Это не независимая оценка и не новые dev результаты.

Каждый inner threshold выбран среди всех различных границ по macro-F1; score>=threshold означает positive, при точной ничьей выбирается наибольший threshold. Thresholds: B7s42=0.37612172961235046, B8s42=0.4373428225517273, B6=0.9390248171985149, B9=0.3443661733578801, C0=0.507323682308197, C1=0.5048826336860657. Полная машинная точность сохранена в prediction metadata; таблицы округлены.

2000 paired bootstrap draws по108 source groups, seed42; одинаковые multiplicities для всех методов. Threshold/model фиксированы в resamples. Это условные percentile95% интервалы без поправки на selection и без учёта variability training seeds. AP — average precision, не трапецеидальная PR-AUC. Работа при calibration precision>=.80 отложена вместе с calibration: не подменяем её inner-настройкой.

## Natural inner-validation

| Method | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC | AP | FPR/FNR | Confusion [[TN,FP],[FN,TP]] |
|---|---:|---:|---|---:|---:|---|---|
| B7s42 | 0.8164 | 0.7959 | 0.6835 / 0.7864 / 0.7314 | 0.8760 | 0.7897 | 0.1697 / 0.2136 | [[367,75],[44,162]] |
| B8s42 | 0.7361 | 0.7061 | 0.5745 / 0.6553 / 0.6122 | 0.7707 | 0.5978 | 0.2262 / 0.3447 | [[342,100],[71,135]] |
| B6 | 0.8009 | 0.7753 | 0.6726 / 0.7282 / 0.6993 | 0.8591 | 0.7016 | 0.1652 / 0.2718 | [[369,73],[56,150]] |
| B9 | 0.7469 | 0.7172 | 0.5905 / 0.6650 / 0.6256 | 0.7784 | 0.6201 | 0.2149 / 0.3350 | [[347,95],[69,137]] |
| C0 | 0.8241 | 0.7950 | 0.7323 / 0.7039 / 0.7178 | 0.8681 | 0.7855 | 0.1199 / 0.2961 | [[389,53],[61,145]] |
| C1 | 0.8225 | 0.7940 | 0.7264 / 0.7087 / 0.7174 | 0.8737 | 0.7809 | 0.1244 / 0.2913 | [[387,55],[60,146]] |

| Method | Macro-F1 95% CI | Positive F1 95% CI | AUROC 95% CI | AP 95% CI |
|---|---|---|---|---|
| B7s42 | [0.7646, 0.8291] | [0.6842, 0.7765] | [0.8434, 0.9077] | [0.7246, 0.8469] |
| B8s42 | [0.6621, 0.7504] | [0.5446, 0.6743] | [0.7251, 0.8121] | [0.5156, 0.6765] |
| B6 | [0.7381, 0.8097] | [0.6416, 0.7500] | [0.8270, 0.8905] | [0.6260, 0.7822] |
| B9 | [0.6703, 0.7596] | [0.5553, 0.6873] | [0.7306, 0.8249] | [0.5434, 0.7045] |
| C0 | [0.7594, 0.8310] | [0.6648, 0.7696] | [0.8342, 0.9001] | [0.7234, 0.8408] |
| C1 | [0.7590, 0.8291] | [0.6649, 0.7689] | [0.8408, 0.9049] | [0.7082, 0.8427] |

Парные разности macro-F1: B7−B8=+0.089812, CI[0.045754,0.135162]; B7−B6=+0.020679, CI[−0.013620,0.056119]; C1−C0=−0.000967, CI[−0.015044,0.013128]; C1−B7=−0.001892, CI[−0.026741,0.020589]. Последние три не доказывают выигрыш. Полные интервалы accuracy/P/R/F1/AUROC/AP/FPR/FNR и CM: reports/phase2b_resume/natural_inner_comparison_cf.json.

B7 — evidence-conditioned ModernBERT span head; B8 — та же модель без question/evidence; B6 — MiniCheck с cleaner/full-context авторского типа; B9 — обученный внутри train length-only prior. C0/C1 определены ниже. Старый TF-IDF dev macro-F1=.5909 относится к ДРУГОМУ split: прямое численное сравнение с этим inner table недопустимо.

## Что обучили и почему

B8: та же архитектура149606402 parameters, answer token targets, label policy, loss, optimizer, train order, seed, microbatch1/accum8 и3epochs, что у B7. Inner42 фиксирован на3epochs, без дополнительного выбора лучшей answer-only эпохи. Full-train42/43/44 служат будущему matched-seed сравнению; они не оцениваются на inner-validation, которую видели при fitting. Три прежних B7 full-train seed не перезапускались. Наличие трёх checkpoints пока НЕ является результатом оценки устойчивости качества по seeds.

C0/C1 инициализированы одним B7 inner42 checkpoint SHA256 7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4. Один fixed epoch, seed42, fresh AdamW lr1e-5/wd.01, warmup10%, BF16/checkpointing, max2048, прежний max-token risk, fixed positive loss weight3.370202392225359. C0:2568 natural +856 label-balanced natural replay. C1:2568 natural +856 повторений120 CF combinations от30 вручную проверенных statements/25 inner-fit sources. В обоих3424 rows,1217 positive/2207 negative, одинаковые optimizer steps.

CF: для однозначного утверждения с целым A строятся S(A)/S(2A); в evidence оба числа присутствуют, второе — отдельный record identifier. Все четыре сочетания premise/answer сбалансированы; только несовпавшее число ответа получает span target. Каждый фиксированный ответ и evidence встречается с обоими labels. Нет ложной NLI-разметки из response labels, произвольной подмене документов не присваивается gold. Это synthetic faithfulness к построенному premise, а не заявление об истинности изменённого факта в мире.

Ограничение изоляции: C0/C1 совпадают по числу строк, response-class balance, шагам и optimizer, но не по длине, домену и token-class balance. Target tokens positive/negative: C0=50632/508039, C1=33688/392113; средняя длина ответа718.07 против548.51chars. Поэтому эффект нельзя приписать исключительно принципу paired training. Узкая синтетика преимущественно про количества/длительности; один date parent. Более строгий same-source двухдатный поиск дал всего3 источника и был отвергнут до обучения.

## Evidence sensitivity и контекст

Авторские300 проверяемых rows:15 underlying facts,5families(date,quantity,unit,negation,attribution),75 fixed-answer supported/contradicted pairs, padding0/8/32 correct sentences, короткий ошибочный claim в начале/конце. Correct paraphrases и перестановки evidence отдельны. Ответ и вопрос в паре одинаковы; обе сущности/числа уже есть в обоих contexts. Это маленькая stress suite, не репрезентативный benchmark и не обучающие данные.

| Method | Mean response Δ contrad−supported | Claim-local Δ | Полностью правильно разделённые пары при natural threshold |
|---|---:|---:|---:|
| B7 | .005153 | .012106 | 0/75 |
| B8 | 0 | 0 | 0/75 |
| B6 MiniCheck | .625550 | .625550 | 30/75 |
| C0 | −.003285 | .009828 | 0/75 |
| C1 | .019626 | .025785 | 2/75 |

У C1 quantity Δ=.07958, date=.00769, unit=−.00173. Mean supported/contradicted risk C0=.16306/.15977, C1=.07557/.09519: улучшение разности сопровождается снижением абсолютного риска даже на противоречиях. Это ещё не работоспособная числовая проверка. На75 supported contexts C1 false alarms0, на75 contradicted находит2; correct paraphrase FP4, reordered correct evidence FP18. C0 соответственно4/3/3/24. У B6 supported FP0, contradictions detected30; высокую natural threshold нельзя менять по этим toy labels.

B7 padding supported/contradicted means:0sent=.23551/.25272;8=.06299/.06684;32=.11772/.11814. Эффект немонотонный. При32 fillers claim-local Δ=.01842, response Δ=.000423: max маскирует небольшой локальный сигнал, но и сам локальный сигнал слаб. Поэтому в R1 выбран supervision pilot, а не перебор aggregators. Перестановка корректного evidence меняет B7 mean risk на.18850 — намного сильнее ошибки отношения.

На32 deterministic inner sources: B7 mean Δ для blank=.46851, cyclic unrelated=.46319, duplicated correct=.07743; C1=.48478/.47913/.09621; B8 во всех случаях ровно0. Модель использует общую совместимость evidence, но этого недостаточно для точных отношений. Arbitrary substitutions здесь не получают новых labels. Исторические B2/B7 sensitivity на32 train sources выполнены отдельно; B7 full-train видел их, поэтому это только mechanistic check, не generalization.

## Fit и перенос синтетики

На120 уникальных CF TRAINING rows (60/60,25sources) B7 macro-F1=.69912, C0=.66510, C1=1.00000; C1 также tokenF1/AUROC/AP=1 и правильно разделяет60/60 fixed-answer pairs. Это IN-SAMPLE diagnostic. Он исключает объяснение «сигнал вообще не оптимизировался», но не отличает запоминание от переносимого узкого правила.

Probe выполнен после завершения seed44,24 октября:28rows/7sources,14negative/14positive,14 fixed-answer pairs. Все источники отсутствуют в supervised fit; exact normalized parent statements отсутствуют в inner-fit evidence. Девять candidates reviewed до predictions; приняты7, два отклонены за approximate/upper-bound wording. B8 добавлен по прямому уточнению пользователя до inference; исходная версия staged сохранена.

| Method | Accuracy | Macro-F1 | Positive P/R/F1 | AUROC / AP | FPR / FNR | Correct pairs | Token / char / exact-span F1 |
|---|---:|---:|---|---|---|---:|---|
| B7 | 0.6071 | 0.6026 | 0.6364 / 0.5000 / 0.5600 | 0.6786 / 0.6928 | 0.2857 / 0.5000 | 3/14 | 0.0984 / 0.0256 / 0.0000 |
| B8 | 0.5000 | 0.3333 | 0.0000 / 0.0000 / 0.0000 | 0.5000 / 0.5000 | 0.0000 / 1.0000 | 0/14 | 0.0000 / 0.0000 / 0.0000 |
| C0 | 0.6071 | 0.5942 | 0.6667 / 0.4286 / 0.5217 | 0.6888 / 0.6736 | 0.2143 / 0.5714 | 3/14 | 0.0727 / 0.0213 / 0.0000 |
| C1 | 0.9643 | 0.9642 | 1.0000 / 0.9286 / 0.9630 | 1.0000 / 1.0000 | 0.0000 / 0.0714 | 13/14 | 0.9630 / 0.9756 / 0.9630 |
| B6 | 0.8214 | 0.8155 | 1.0000 / 0.6429 / 0.7826 | 1.0000 / 1.0000 | 0.0000 / 0.3571 | 9/14 | N/A |

C1=27/28correct, CM[[14,0],[1,13]], mean evidence delta=.868688; C0=.191750, B7=.151118, B8=0, B6=.602685. Macro-F1 CI C1[.891613,1], C0[.426901,.749681]; paired C1−C0=+.370037,95%CI[.214559,.558650] по2000 одинаковым source draws. Семь parent groups и один преобразователь ограничивают интерпретацию этих интервалов.

C1 исправил10 ошибок C0 и10 B7, новых относительно них0. Относительно B8 исправил13; относительно B6 исправил5, но добавил1. Единственный C1 FN — synthetic source14364, замена указанной длительности3→6years: C1risk=.228501 при threshold=.504883; B6risk=.980281. На13 остальных positives ошибочное число выделено точно, без дополнительных FP tokens. Числовое значение нормативной длительности трактуется generator как заданное; возможные pragmatic/minimum readings остаются ограничением synthetic semantics, gold не менялся.

Дополнительный paired2000 source bootstrap C1−B6 macro-F1: +.148693,95%CI[−.036592,.358464], reports/phase2b_resume/cf_unseen_external_bootstrap.json. B6 также имеет AUROC/AP=1; отличие его classification от C1 зависит от заранее выбранного operating point. Не подбираем новые пороги на probe. Нельзя назвать это преимуществом над MiniCheck на общей задаче. C1 переносится на новые parent facts в ТОМ ЖЕ формате; это против объяснения только запоминанием конкретных training statements, но совместимо с узким template-specific matching. Большая authored suite остаётся слабой, natural inner улучшения нет.

Probe inference only, без model load/bootstrap: B7=1.698s/16.49responses/s, B8=1.152s/24.30, C0=1.291s/21.69, C1=1.141s/24.54, B6=1.600s/17.49. PeakCUDA span≈.886GB,B6≈2.454GB. Process RSS maxima в одном последовательном процессе накапливаются и не являются независимым memory benchmark методов. Full predictions/runtime/CM/span metrics: reports/phase2b_resume/cf_unseen.json и ignored predictions.


## Локализация и ошибки

Custom micro token/character/exact-spanF1 при token threshold.5: B7=.58489/.58862/.03540; B8=.33054/.33923/.00128; C0=.57428/.57662/.04180; C1=.58193/.58437/.03748. Token target=любое пересечение с gold character span; char scoring исключает whitespace; exact spans обрезают краевые whitespace и объединяют пересекающиеся/соседние через whitespace ranges. Это наш scorer, НЕ официальный RAGTruth scorer.

23 positive answers>=500chars с total gold length<=5%: response detection B7=17,B8=16,B6=15,C0=15,C1=15. При .5 token threshold есть gold-overlap token у B7=11,C0=9,C1=9. Эти числа нельзя напрямую вычитать из response detections: thresholds разные. Дополнительная проверка при ОДНОМ и том же response threshold: B7 обнаруживает17, gold-token hit13, detection elsewhere4; B8=16/7/9; C0=15/9/6; C1=15/9/6. Any overlap — слабое условие, не полное восстановление спана. Подгруппы малы; нет вывода о генеральной частоте ошибок.

B7 FPR по answer tokens:<64=.01389,64–128=.10345,128+=.28571; у B8 для128+=.39796. В первой группе всего4positive из76. Correlation с длиной не доказывает причинный механизм. Среди75 B7 FP только7 имеют punctuation peak,64 имеют>=5 above-threshold tokens, ни одного only-punctuation FP. Простая маска пунктуации не устраняет основной bottleneck.

Типы gold errors (positive-only overlapping groups): B7 recall EvidentBaseless106/118,SubtleBaseless63/69,EvidentConflict33/59,SubtleConflict6/11; C1=97/118,59/69,29/59,4/11. В training conflicts203/1292spans и13968/152360 annotatedchars:15.71% и9.17%. Это наблюдение о составе targets, не измерение доли gradient. Полные token/sentence/generator strata в natural_inner_comparison_cf.json; generator ID не является inference feature.

Сохранены все8 исправленных и9 новых C1 ошибок относительно C0. Ручной audit двух самых больших score changes каждого направления:

- 13628: C1 обнаруживает неподдержанный дополнительный cooking step/10minutes, peak попадает в gold; C0=.41703→C1=.94122. B7 уже был прав: это восстановление ошибки C0, не новый выигрыш над B7.
- 15188: C0=.46478→C1=.77999, binary answer исправлен, но max token вне gold span. Не засчитываем как доказательство корректной локализации.
- 17393: новый FP на поддержанном сравнительном paraphrase; .28062→.67512.
- 15746: новый FP на корректной numeric/unit инструкции из evidence; .29890→.63714.

Контекстный audit B7/B8:103 B7-correct/B8-wrong против51 reverse.15332/14330 иллюстрируют unsupported additions, которые evidence помогает отвергнуть;15960 — FP на paraphrase/inference.17315 — корректно помеченная hypothetical arithmetic illustration: изменённые числа сами по себе не означают hallucination. Никаких ID exceptions, пересмотра gold или data deletion. Полные тексты, evidence, offsets и predictions находятся только в ignored predictions/phase2b_resume. Дополнительно error_export_R1.json сохраняет15 самых высокорисковых FP и15 самых низкорисковых FN каждого B7/B8/C0/C1; это целевой разбор, не случайная выборка.

## Ресурсы и исполнившиеся опыты

RTX3060 12GB; Windows, Python3.12.14, PyTorch2.8.0+cu128, pinned environment. Один GPU-heavy worker за раз, без paid/cloud. B8 maximum input491tokens; contextual QA максимум1008 при limit2048,100% answer/evidence coverage. Нет молчаливого сокращения контекста; long-context за пределами этих наблюдённых inputs не доказан.

| Fit | Train rows/sources | Epochs | Seconds | Peak CUDA allocated bytes |
|---|---|---:|---:|---:|
| B8 inner42 | 2568/428 | 3 fixed | 1483.2450 | 3162059264 |
| B8 full42 | 3216/536 | 3 fixed | 1669.0719 | 3162160128 |
| C0 inner42 | 3424/428 | 1 | 727.3860 | 3162549248 |
| C1 inner42 | 3424/428 | 1 | 696.7204 | 3162549248 |

B8 full43:1717.6895s, peakCUDA3162157056bytes; full44:1709.1482s,3161230336bytes. Оба завершены3epochs,3216rows/536sources. После ошибки capacity была проверена живая очередь: обучение продолжалось в фоне и завершилось без повторного запуска. Все шесть новых fits (B8full42/43/44,inner42,C0,C1) проверены по config/history/weights/tokenizer; SHA256 в completed_fits_verified.json.

Resident-model natural inner scoring,648responses: B7=30.099s/21.53responses/s/925.6MB allocated; B8=29.171s/22.21/894.3MB; B6=148.018s/4.38/2985.5MB; C0=31.131s/20.82/925.6MB; C1=31.002s/20.90/925.6MB. Amortized response latency respectively46.45/45.02/228.42/48.04/47.84ms. B6batch4/4421 claim-document pairs, neural spanbatch1. These are one-run wall timings, not production latency SLA. B7 inclencoding30.898s, B8=29.504s. B9 head-only timing excludesfeatureconstruction and is not comparable end-to-end. Allocated CUDA is not total process/driver memory. Process peak RSS scoring: B7=1.603GB, B8=1.605GB, B6=3.652GB, C0=1.581GB, C1=1.581GB. B8 full43 peakCUDA3162157056bytes, peakRSS2972123136bytes.

Суммарное измеренное время шести новых fits=8003.260951s (133.39min); это время fits, не длительность всей управляющей сессии. Реальные smoke/profile прошли до training. R1 OOM/training failures не было. Старые Phase2B profile failures сохранены в исторических отчётах. CPU pytest initially failed due sandbox temp permissions; rerun с workspace basetemp решил проблему. Первый optional-data запуск имел неверную environment variable и пропустил test; это не выдаётся за полный check.

## Изоляция, происхождение и ограничения

Прежние92 pause-manifest files проверены SHA256 перед началом. Старые68 Phase2A snapshot files ранее восстановлены с побайтно одинаковыми dev predictions; новые source hashes не подставляются в старый freeze. Raw files и outer membership сохраняются. Partial normalized659-char passage общий между inner-fit14426 и inner-val15181, а также outer dev15272. Inner supplemental642-row evaluation при прежних thresholds: B7=.79556,B8=.70622,B6=.77613. Основной648-row набор неизменен, никакого test content audit не запускалось.

Backbone ModernBERT revision8949b909ec900327062f0ebf497f51aef5e6f0c8; MiniCheck2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb: нет известного supervised RAGTruth overlap по доступной provenance, нельзя гарантировать чистоту pretraining. HHEM2.1 — недостаточно известная lineage, только отдельная diagnostic. Public RAGTruth-trained LettuceDetect/Guard имеют подтверждённое пересечение с нашим held-out-from-official-train; не включены как clean generalization. Полная таблица checkpoints/licenses/context/memory/code/comparability в docs/PHASE2B_COMPETITORS.md. Нет заявления о SOTA или преимуществе над опубликованными результатами.

Ограничения: inner selection bias; одна inner seed42; узкая синтетика/мало родителей; различия token balance C0/C1; partial passage reuse; toy suite не репрезентативна; отсутствие question-only обученного контроля. Bootstrap не снимает эти ограничения. Future untouched-test/external protocol описан в docs/PHASE2B_CONFIRMATORY_PROTOCOL.md; ничего из него сейчас не запущено.

## Проверки и сохранность

132 tests passed с real-data/GPU opt-ins (61.70s), Ruff passed, pip check passed в обоих окружениях. Исходная .venv:127offline tests passed/2opt-in skipped до probe additions, затем7probe tests passed после интеграции. CI matrix3.11/3.12 сохранена offline; remoteCI не проверен. Raw/membership guards прошли в полном suite; новый detector на dev/test не оценивался.

Повторно совпали все92 old pause artifacts. Перед integration все42 source/config/plan files текущего seed44 совпали с content-addressed snapshot и сохранёнными hashes. В staged оказалось21source file:12 уже применены побайтно,6pending additions,3historical variants. Все21 сохранены в НЕигнорируемом research_snapshots/R1_staged_recovery_20261024.zip с SHA256 manifest и diff; CRC/SHA всех entries проверены. Только6актуальных дополнений интегрированы; старые варианты не накатывались. Добавление B8 в probe отдельно отражено в плане.

Models/weights/raw texts/cache остаются ignored; source recovery ZIP не содержит datasets/weights/secrets. Git commit/push/publishing не выполнялись. Финальные R1 artifact manifest и source snapshot: models/phase2b_resume/R1_artifact_manifest.json, models/phase2b_resume/R1_source_snapshot.zip.

## Воспроизведение

Команды — из корня репозитория в существующем .venv-research. Старые B7 full42/43/44 НЕ переобучать. Это рецепт повторения в отдельной копии с pinned assets и прежним B7 inner checkpoint, но пустыми новыми B8/R1 output paths. Текущие команды записи откажутся перезаписывать результаты. Для точного исторического воспроизведения используйте source snapshot соответствующего run; не отключайте hash guards.

~~~powershell
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage final --seed 42 --batch-size 1 --answer-only
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage inner --seed 42 --epochs 3 --fixed-epochs --batch-size 1 --answer-only
.\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run score --method B8s42 --partition inner_val
# Только после measured budget gate, последовательно:
foreach ($seed in @(43,44)) {
  .\.venv-research\Scripts\python.exe -m grounded_hallucination_lab.research.run train-span --stage final --seed $seed --batch-size 1 --answer-only
}
foreach ($method in @('B7','B8','B6')) {
  .\.venv-research\Scripts\python.exe scripts/research_resume.py controls --method $method
}
foreach ($method in @('B7','B8')) {
  .\.venv-research\Scripts\python.exe scripts/research_resume.py perturb --method $method
}
.\.venv-research\Scripts\python.exe scripts/research_resume.py analyze
.\.venv-research\Scripts\python.exe scripts/research_sensitivity.py B2
.\.venv-research\Scripts\python.exe scripts/research_sensitivity.py B7s42
.\.venv-research\Scripts\python.exe scripts/research_counterfactual.py prepare
.\.venv-research\Scripts\python.exe scripts/research_counterfactual.py profile
foreach ($method in @('C0','C1')) {
  .\.venv-research\Scripts\python.exe scripts/research_counterfactual.py train --method $method
  .\.venv-research\Scripts\python.exe scripts/research_counterfactual.py score --method $method
  .\.venv-research\Scripts\python.exe scripts/research_resume.py controls --method $method
  .\.venv-research\Scripts\python.exe scripts/research_resume.py perturb --method $method
}
.\.venv-research\Scripts\python.exe scripts/research_resume.py analyze-cf
.\.venv-research\Scripts\python.exe scripts/research_cf_fit_diagnostic.py
.\.venv-research\Scripts\python.exe scripts/research_cf_unseen.py
.\.venv-research\Scripts\python.exe scripts/research_r1_error_audit.py

$env:GROUNDCHECK_RAGTRUTH_DIR = 'data/raw/ragtruth'
$env:GROUNDCHECK_RESEARCH_MODELS = '1'
.\.venv-research\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=models/phase2b_resume/pytest_reproduction
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS
.\.venv-research\Scripts\python.exe -m ruff check .
.\.venv-research\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip check
~~~

Перед повторением training обязателен smoke на representative inputs; сохранённый answer-only профиль и скрипт — в R1 snapshot. Очереди/логи в models/phase2b_resume сохраняют фактические args, start times, exit codes и budget gates; их запуск в заполненном workspace не является способом продолжить.

Нет calibration/dev/test команды. Старый majority regression в тестах проверяет established baseline, не новую модель.

## Изменённые файлы

Относительно состояния на паузе (не включая неизменённые, но отсутствующие в старом ZIP файлы и generated egg-info):
- src/grounded_hallucination_lab/research/{run.py,span_model.py,protocol.py,controls.py,counterfactual.py}.
- scripts/{research_inner_audit.py,research_resume.py,research_counterfactual.py,research_cf_fit_diagnostic.py,research_cf_unseen.py,research_r1_error_audit.py}.
- tests/{test_research_resume.py,test_counterfactual.py}.
- configs/{counterfactual_acceptance.json,counterfactual_validation_acceptance.json}.
- EXPERIMENT_PLAN_RESUME.md, EXPERIMENT_R1_CF.md, EXPERIMENT_R1_DIAGNOSTIC.md, RESEARCH_NOTES.md.
- README.md, docs/{PROGRESS.md,DECISIONS.md,EVALUATION_PROTOCOL.md,PHASE2B_R1_RESULTS.md}.
- research_snapshots/R1_staged_recovery_20261024.zip и соответствующий JSON manifest.
- Новые numeric reports в reports/phase2b_resume; local checkpoints/logs/predictions/source snapshots — в ignored models/ и predictions/.

Новые runtime guards: fixed inner epochs явны; final fits проверяют полный train membership/seed/config/history и отсутствие held-out selection; leakage tests запрещают test/withheld и fitted source parents в probe. Training generator не менялся при интеграции probe. Зависимости, старые datasets, partition policy, исходный EXPERIMENT_PLAN.md и исторические freeze hashes не переписаны.

## Следующий пакет

1. **Перенос между форматами и контекстами.** Гипотеза: C1 научился полезному, но слишком узкому numeric matching. Внутри train заранее отделить source/template groups; расширить verified pairs до естественных многофактных контекстов и положительных paraphrases. Контроль — тот же объём/длины/числовые частоты без paired relation augmentation. Главный критерий — transfer на unseen templates и natural inner качество/localization, а не training-fit. Не добавлять entity/negation все сразу.
2. **Изолированный paired loss.** На ОДНИХ И ТЕХ ЖЕ данных предыдущего опыта сравнить обычный token CE с дополнительным margin для одной answer claim при supported/unsupported evidence. Проверить, усиливается ли claim-local evidence delta без подъёма FP на корректных paraphrases. Backbone/retrieval/aggregation фиксированы. Это известная contrastive идея, не заявление о новизне.
3. **Контекст вокруг проверяемого фрагмента.** Сначала read-only extractive claim+neighbor window ablation на фиксированном encoder с точными offsets; затем matched training только если диагностика показывает, что длинный ответ размывает локальный сигнал. Контроль — прежний полный ответ. Не генерировать новые claims, не терять факты внутри отказов, учитывать coreference и short-error recall. Max/length-robust pooling рассматривать после проверки локального сигнала, а не вместо неё.

Каждый опыт требует отдельного preregistered плана, semantic gates и реального profiling. Нынешний continuation epoch≈12min/3.163GB — ориентир только для нынешних inputs, не обещание для нового контекста. Перспективный результат необходимо повторить минимум на трёх inner training seeds; текущие full-train B7/B8 seeds43/44 не заменяют такую проверку. Question-only matched training остаётся отдельным нерешённым контролем атрибуции. Нового dev batch не будет до согласования стабильного протокола; test и внешний confirmatory набор остаются закрытыми.
