# GroundCheck — verdict bounded discovery loop, 2026-10-26

Четыре цикла завершены; исследование продолжается, release не готовился. Рассмотрены 8 гипотез из 4 семейств. Проверены 4 основные гипотезы, дополнительно выполнены annotation/provenance audit и однократная внешняя confirmation.

**Согласованного улучшения natural detection, factual sensitivity и localization нет.** B7 остаётся лучшим из этих методов на reused RAGTruth inner. FactCG существенно сильнее на factual interventions и переносится лучше на внешнюю QA-задачу. Это результат готовой модели и нового сравнения, не собственная новая архитектура.

## Что реально проверено

| Цикл | Проверка | Результат и решение |
|---|---|---|
|1/H1|Max только по alphanumeric tokens, те же probabilities|68 natural scores изменились; решений 0, ΔMF1 CI[0,0]. Гипотеза существенного улучшения отвергнута.|
|2/H2|Pinned FactCG, author evidence packing + extractive claims|Контролируемые связи различает сильно; natural FP и coarse localization провалили replacement gate.|
|3/H5|96 training-only FP32 forwards: actual logit-gradient mass|Для 32 коротких ошибок медианная positive share=.4002, не<.20. Gate закрыт; backward/profile/fit не запускались.|
|4/H3|Тот же FactCG, весь answer вместо clean fragments|FP меньше, FN значительно больше; короткие ошибки 18→13/19. Gate провален; никаких пороговых исправлений.|

Cycle3 treatment увеличил бы logit-gradient share до.8764, сохранив negative coefficients и суммарный positive coefficient 558.7074; это расчёт на frozen logits, не результат обучения. Малое число error tokens не означает малый gradient: трудные positive tokens дают существенный сигнал. Encoder-gradient interference этим не проверена.

## Natural development: одинаковые 642 ответа /107 sources

438 negative / 204 positive; include/all. Это многократно использованный inner, не независимое подтверждение. Состав R9/R10 сохранён: source 15181 был исключён ранее по shared-passage правилу; historical 648 и outer partitions не менялись.

|Метод|Accuracy|Macro-F1 [95%CI]|Positive P/R/F1|AUROC|AP|FPR/FNR|Token F1|
|---|---:|---|---|---:|---:|---|---:|
|B7|0.8162|0.7956 [0.7633, 0.8273]|0.6838/0.7843/0.7306|0.8759|0.7897|0.1689/0.2157|0.5870|
|FactCG segmented|0.6604|0.6592 [0.6159, 0.6979]|0.4825/0.9461/0.6391|0.8814|0.7694|0.4726/0.0539|0.4074|
|FactCG whole|0.6636|0.6305 [0.5866, 0.6731]|0.4756/0.5735/0.5200|0.6685|0.4671|0.2945/0.4265|0.2188|

Confusion matrices (true rows0/1): B7 [[364,74],[44,160]]; segmented [[231,207],[11,193]]; whole [[309,129],[87,117]]. Short TP / gold-token-hit из 19: B7 14/8, segmented18/14, whole13/13. Whole помечает весь answer, поэтому его hit не является точной локализацией. Character F1 и exact-span F1, denominators и все raw counts сохранены в cycle2_analysis.json/cycle4_analysis.json. Scorer собственный: token any gold-overlap, nonwhite characters, exact trimmed merged spans приtoken>=.5; не официальный RAGTruth scorer.

Segmented−B7 ΔMF1=−.13631,95%[−.18699,−.09032]. Whole−segmented ΔMF1=−.02874 [−.08083,.02741]; ΔFPR=−.17808[−.24243,−.11899], ΔFNR=+.37255[.29523,.45002]. Объединение shared-passage sources 14325/15545 в 106 clusters даёт ΔMF1 [−.08302,.02833], без смены вывода. Bootstrap2000/seed 42 парный поsource, не устраняет adaptive research bias и не заменяет training seeds.

## Factual interventions: synthetic, отдельная постановка

|Suite/метод|N/parents|Macro-F1|Both-correct pair accuracy|Signed claim/slot gap|
|---|---|---:|---:|---:|
|r1/B7|150/15|0.3680|0.0000|0.0105|
|r1/segmented|150/15|1.0000|1.0000|0.7774|
|r1/whole|150/15|0.9733|0.9467|0.5594|
|r6/B7|768/16|0.4399|0.0182|0.0081|
|r6/segmented|768/16|0.9635|0.9271|0.5160|
|r6/whole|768/16|0.9609|0.9219|0.5012|

Same-answer evidence worlds: это чувствительность к конкретным отношениям, не blank-evidence proxy. Все варианты одного parent остаются одним bootstrap cluster. R6 segmented−B7 local-gap CI[.47902,.53340]. Whole−segmented R1 pair Δ=−.05333 [−.10667,−.01333]. Успех здесь не переносится автоматически в natural.

Safe R10 order subset204/34: any-rotation flips B7=18, segmented=24, whole=21; mean absolute risk Δ=.04204/.04870/.04344. Улучшение semantic discrimination не сопровождалось большей invariance. Ни лучший rotation, ни новый threshold не выбирались.

## Внешняя human-triplet-derived оценка

RefChecker/Dolly source split40dev/60locked заморожен до labels. До confirmation исключены 6 источников одинаково для всех:4070 за training-text overlap FactCG и 5 за complete-evidence capacity B7. Dev274/40 (82 positive); confirmation363/54 (104 positive/259 negative).15 empty-triplet answers excluded as unknown вconfirmation. Human labels относятся к Claude2-extracted triplets; response label=any Neutral/Contradiction. Это не exhaustive response gold, никаких external span labels не выдумывали.

|Раздел/метод|N/sources|Macro-F1|Positive F1|FPR|Recall+|ΔMF1 vsB7 95%CI|
|---|---|---:|---:|---:|---:|---|
|development/all/B7|274/40|0.6147|0.4487|0.2031|0.4268|reference|
|development/all/FactCG|274/40|0.7054|0.6087|0.2396|0.6829|[0.0190, 0.1622]|
|development/all/Whole|274/40|0.7128|0.5976|0.1719|0.5976|[0.0220, 0.1660]|
|confirmation/all/B7|363/54|0.6338|0.4956|0.2548|0.5385|reference|
|confirmation/all/FactCG|363/54|0.7037|0.6066|0.2548|0.7115|[0.0030, 0.1441]|
|confirmation/all/Whole|363/54|0.7258|0.6140|0.1737|0.6346|[0.0247, 0.1628]|
|confirmation/closed_qa/B7|273/40|0.5932|0.4204|0.2587|0.4583|reference|
|confirmation/closed_qa/FactCG|273/40|0.7137|0.5963|0.2040|0.6667|[0.0469, 0.2050]|
|confirmation/closed_qa/Whole|273/40|0.7359|0.6111|0.1393|0.6111|[0.0658, 0.2252]|
|confirmation/information_extraction/B7|59/9|0.7266|0.6061|0.1628|0.6250|reference|
|confirmation/information_extraction/FactCG|59/9|0.6437|0.5652|0.3953|0.8125|[-0.1690, -0.0070]|
|confirmation/information_extraction/Whole|59/9|0.6745|0.5641|0.2791|0.6875|[-0.1619, 0.0473]|
|confirmation/summarization/B7|31/5|0.6688|0.7222|0.4667|0.8125|reference|
|confirmation/summarization/FactCG|31/5|0.6314|0.7027|0.5333|0.8125|[-0.4556, 0.4145]|
|confirmation/summarization/Whole|31/5|0.6771|0.6875|0.3333|0.6875|[-0.2482, 0.3229]|

Preregistered primary external gate **PASS**. FactCG segmented−B7 ΔMF1=+.06991[.00302,.14408], FPR одинаковый 66/259, TP74 вместо 56/104. Whole выше по point estimate, но это заранее объявленная secondary ablation; не выбираем его победителем после confirmation. ClosedQA secondary:273/40, FactCG .71372 vsB7 .59322. Information extraction ухудшается:59 ответов/9 sources; summarization31/5 слишком мала для надёжного общего вывода. Category comparisons без multiplicity correction. Теперь confirmation использована: для следующего нового утверждения нужна новая untouched оценка.

## Ошибки, confounds и annotation quality

В SHA-selected12/147 новых FP (11 sources), provisional assistant audit:4meta/absence/passage-reference,2courtesy/advice,3instructions/causal paraphrases,3possible annotation omissions. Это не independent human review и не оценка noise rate. Например, в 12563 высокий риск у courtesy intro;16836 — causal safety paraphrase с evidence;16366 меняет P45 на Tax Return Copy, хотя gold0. Все gold сохранены. Whole подавляет 6/12 тревог, но не подавляет 3suspected omissions. Lost short TP:11937,11961,15663,15829,17596.

Важный post-run confound: whole возвращает 35 pure meta/refusal answers, которые clean_segments полностью исключал; все 35 становятся FP. Поэтому C4 меняет также checkworthiness filtering, и не изолирует только discourse. Это обнаружено и задокументировано, без исправления и повторного подбора после результата. Только 6natural ответов/1 source имеют 2evidence chunks: общие выводы о distributed support невозможны.

17 mixed-span repeated-sentence groups/12 sources не равны 17label errors: heading и passage attribution меняют смысл. Automatic span propagation отвергнута. Annotation quality и detector quality нельзя объединять в одну метрику.

## Стоимость и проверки

Conservative charged GPU-process wall **24.59/120 min**, включая все research tests и model-loading intervals. Новых fit/checkpoints 0; скачан один 435M FactCG (1,740,304,440 bytes), а не обучена новая модель. Все FactCG inputs<=908 tokens в primary runs; confirmation<=820. Zero truncated pairs. Пиковая measured allocated VRAM FactCG 2.02 GiB; per-job latency/throughput/RSS/VRAM в *_resources.json. Cache reused exact evidence/claim pairs; timings нельзя выдавать за отдельный uncached benchmark каждого метода. Энергия/CUDA-active time не измерялись.

Offline pytest 282 passed/8 skipped; research opt-in289 passed/1 skipped. Последнийskip — реальные outer-data regression не разрешены. Ruff проходит с одной явной E501 waiver для 102-character metadata line в immutable confirmation helper; helper/hash не менялись послеfreeze. Оба pip check: no broken requirements. CI файл tracked, matrix 3.11/3.12 offline; remote CI не запускался/не подтверждён. Historical verification: R9 136 artifacts,R10 120 artifacts,B7/R10A/R10B checkpoints и oldsource guards PASS.

## Происхождение, novelty и следующий шаг

FactCG HF 0430e3509dbd28d2dff7a117c0eae25359ff3e80; author repo 41f185413312ed9b3904ec03963fe899693498e0, MIT, standard Transformers без custom remote code. Нет известного declared RAGTruth supervised overlap; доступные 6 training files не совпали 16 wordwindows с allowed R10 train/inner. Но 5391 MuSiQue premises отсутствуют в release, backbone pretraining непроверяемо. RefChecker source 4070 исключён доlabels; это ограниченный audit, не гарантия отсутствия contamination. Primary literature и 8 hypotheses — RESEARCH_DISCOVERY_LOOP.md; все gates — R11_PLUS_RESEARCH_LOG.md и отдельные preregs.

**Прорыва и доказанной научной новизны нет.** Есть честно подтверждённый external transfer gain готового verifier и опровержение двух простых remedies. Нельзя объявлять generic factual relation reasoning недоступным маленьким моделям: FactCG уже решает эти контрфактические пары. Нельзя обещать, что ещё один synthetic template или CE loss исправит natural detection: наши данные этого не поддерживают.

Следующая фундаментальная задача — согласовать **что является проверяемым утверждением**, evidence-conditioned verdict и точную локализацию. Наиболее информативный следующий пакет: на новых train-side natural источниках отдельно оценить assertion/refusal/absence-policy selection, затем фиксированный FactCG verifier и B7 localization без изменения порогов. Gold factuality и checkworthiness нужны независимо, а не автоматически из нашего extractor. Три последующих проверки: (1) source-disjoint public human claim labels с контролем полноты extraction; (2) matched factual/nonfactual scope controls без выбрасывания фактов внутри отказа; (3) отдельный evidence-conditioned span objective только если первые две проверки покажут конкретный supervised signal. Эти опыты здесь не запускались.

Команды, hashes, snapshot и полный перечень файлов: DISCOVERY_REPRODUCE.md. RAGTruth outer calibration/dev/test не открывались; публикации и release нет.
