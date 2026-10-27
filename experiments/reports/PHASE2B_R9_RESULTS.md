# R9 — Natural passage competition: результат отрицательный, механизм уточнён

2026-10-25. Исследовательский пакет завершён; research продолжается. Нового обучения,
outer calibration/dev/test, публикации или release нет.

**Основной результат после исправления alignment:** independent passage verification
с выбором одного passage на предложение ухудшает B7 macro-F1 с .795557 до .627247;
Δ95% [−.210954, −.124550]. R6-B: .783628→.703526, Δ95% [−.121661, −.039404].
Gate не пройден. Ранее напечатанные .53131/.57453 относятся к ошибочному pooling v2
и **не являются научным результатом**. Все таблицы ниже — final/v4.

## Вопрос, литература и протокол

До новых predictions записан `R9_RESEARCH_PROPOSAL.md` (historical local artifact; not distributed):
четыре проверенных первоисточника (RAGTruth, MiniCheck, AlignScore, LettuceDetect),
три конкурирующие гипотезы, критерии успеха/отказа и ресурсный gate.
Не повторяли R1 blank/unrelated-evidence ablation: широкая evidence sensitivity
там уже показана. Проверяли более узкий вопрос: мешает ли совместная обработка
нескольких passages естественному grounding и можно ли безопасно выбрать поддержку
локально, обходясь без графового extractor.

642 ответа / 107 source groups,438 negative/204 positive, policy include/all.
Это **ранее многократно использованный natural inner-validation**, не независимый
benchmark. Исключены только6 ответов известного shared-passage source15181 по
предварительному правилу; исходные partitions и исторические648 результатов не меняли.
Источники не пересекаются с supervised train обоих checkpoints. Весь пакет train-side.

B7 inner42 и R6-B42 frozen, FP32/batch1/TF32off, max_length2048, response risk=max token
probability, threshold>=.37612172961235046; localization cutoff .5. Полный answer и
question остаются в каждом forward. Ни gold spans, ни generator/quality metadata
не входят в inference features. Веса, thresholds и исходные predictions неизменны.

Шесть реальных neural условий на модель: full; циклические перестановки трёх целых
passages на1/2; три passages по отдельности. Перестановки сохраняют исходные ID,
тексты и внутренний порядок каждого passage. Никаких новых фактов или удаления
контекста в rotations. Полный answer покрыт всегда, truncation0; pooling видит
объединение всех исходных passages через три forwards.

Из single-passage probabilities получены два заранее запланированных варианта:
- **sentence_pool:** для каждого extractive Punkt-предложения выбрать passage с
  минимальным максимальным token risk, перенести все его token scores для этого
  предложения; response score по-прежнему max. Полный answer остаётся в нейросети.
- **token_min:** минимум риска по passages отдельно для каждого токена, затем max
  ответа. Это permissive механистический контроль: потенциально может склеивать
  несовместимую поддержку. Не выдаётся за доказательство истинности отношений.

Single-passage inputs не получили новых gold labels: их F1 не вычисляли. Original
labels применяются к full/rotations и конечным pooling-системам, получающим всё evidence.
По ним не подбирался threshold. Один фиксированный seed/checkpoint, без best-seed выбора.
Парный percentile source bootstrap2000, seed42, одинаковые draws у всех методов.
CI условны на фиксированных моделях; не исправляют reused-validation/selection bias,
не оценивают variability training seeds и не скорректированы за multiple comparisons.

## Response-level результаты

MF1 — macro-F1, F1+ — hallucination-class F1, AP — Average Precision, не trapezoidal PR-AUC.
Во всех строках одинаковые642 IDs/107 sources и204 positives. MF1 CI — абсолютный95%.

| Model / arm | Accuracy | MF1 [95%] | F1+ | P+ | R+ | AUROC | AP |
|---|---:|---|---:|---:|---:|---:|---:|
| B7/full | 0.8162 | 0.7956 [0.7633,0.8273] | 0.7306 | 0.6838 | 0.7843 | 0.8759 | 0.7897 |
| B7/rotate1 | 0.8178 | 0.7980 [0.7658,0.8314] | 0.7347 | 0.6835 | 0.7941 | 0.8767 | 0.7964 |
| B7/rotate2 | 0.8193 | 0.8016 [0.7664,0.8345] | 0.7422 | 0.6789 | 0.8186 | 0.8787 | 0.7929 |
| B7/sentence_pool | 0.6277 | 0.6272 [0.5791,0.6759] | 0.6139 | 0.4578 | 0.9314 | 0.8458 | 0.7404 |
| B7/token_min | 0.6449 | 0.6435 [0.5964,0.6896] | 0.6213 | 0.4698 | 0.9167 | 0.8666 | 0.7857 |
| R6B/full | 0.8100 | 0.7836 [0.7485,0.8171] | 0.7081 | 0.6916 | 0.7255 | 0.8729 | 0.7830 |
| R6B/rotate1 | 0.8240 | 0.8003 [0.7648,0.8343] | 0.7316 | 0.7097 | 0.7549 | 0.8756 | 0.7896 |
| R6B/rotate2 | 0.8162 | 0.7902 [0.7506,0.8266] | 0.7163 | 0.7028 | 0.7304 | 0.8740 | 0.7853 |
| R6B/sentence_pool | 0.7087 | 0.7035 [0.6608,0.7455] | 0.6643 | 0.5241 | 0.9069 | 0.8404 | 0.7169 |
| R6B/token_min | 0.7383 | 0.7288 [0.6893,0.7678] | 0.6782 | 0.5566 | 0.8676 | 0.8717 | 0.7840 |

| Model / arm | CM [[TN,FP],[FN,TP]] | FPR | FNR |
|---|---|---:|---:|
| B7/full | [[364, 74], [44, 160]] | 0.1689 | 0.2157 |
| B7/rotate1 | [[363, 75], [42, 162]] | 0.1712 | 0.2059 |
| B7/rotate2 | [[359, 79], [37, 167]] | 0.1804 | 0.1814 |
| B7/sentence_pool | [[213, 225], [14, 190]] | 0.5137 | 0.0686 |
| B7/token_min | [[227, 211], [17, 187]] | 0.4817 | 0.0833 |
| R6B/full | [[372, 66], [56, 148]] | 0.1507 | 0.2745 |
| R6B/rotate1 | [[375, 63], [50, 154]] | 0.1438 | 0.2451 |
| R6B/rotate2 | [[375, 63], [55, 149]] | 0.1438 | 0.2696 |
| R6B/sentence_pool | [[270, 168], [19, 185]] | 0.3836 | 0.0931 |
| R6B/token_min | [[297, 141], [27, 177]] | 0.3219 | 0.1324 |

| Сравнение с full того же checkpoint | Δ macro-F1 | Парный95% CI |
|---|---:|---|
| B7/rotate1 | +0.0024 | [-0.0153,+0.0204] |
| B7/rotate2 | +0.0060 | [-0.0159,+0.0277] |
| B7/sentence_pool | -0.1683 | [-0.2110,-0.1245] |
| B7/token_min | -0.1521 | [-0.1911,-0.1119] |
| R6B/rotate1 | +0.0167 | [-0.0014,+0.0347] |
| R6B/rotate2 | +0.0066 | [-0.0108,+0.0234] |
| R6B/sentence_pool | -0.0801 | [-0.1217,-0.0394] |
| R6B/token_min | -0.0548 | [-0.0878,-0.0206] |

R6-B full−B7 full: ΔMF1−.011929,95%[−.030730,+.006547]. Нельзя объявлять разность
macro-F1 доказанной; нельзя автоматически заменять B7 моделью R6-B. Его более высокий
synthetic score не является основанием для natural selection.

## Локализация, длина и механизм

Custom scorer, **не официальный RAGTruth**: token target — любой gold-char overlap;
character F1 — non-whitespace characters; exact spans — trimmed ranges, merged при
пересечении/только whitespace gap. Gold spans используются только в анализе.
Short subset:19 positive answers /16 sources,>=500 chars, gold nonspace<=5% answer.
Response TP и попадание в gold token при .5 — разные события.

| Model / arm | Token F1 | Char F1 | Exact-span F1 | Short TP /19 | Short gold-hit /19 |
|---|---:|---:|---:|---:|---:|
| B7/full | 0.5870 | 0.5912 | 0.0379 | 14 | 8 |
| B7/sentence_pool | 0.5300 | 0.5336 | 0.0339 | 19 | 14 |
| B7/token_min | 0.5601 | 0.5630 | 0.0317 | 18 | 13 |
| R6B/full | 0.5802 | 0.5827 | 0.0558 | 13 | 5 |
| R6B/sentence_pool | 0.5354 | 0.5389 | 0.0325 | 18 | 12 |
| R6B/token_min | 0.5741 | 0.5766 | 0.0360 | 16 | 12 |

Sentence_pool−full token-F1 CI: B7[−.08906,−.02298], R6-B[−.07924,−.00958].
Short gold-hit улучшился8→14 у B7 (ΔCI[.125,.545]) и5→12 у R6-B
([.167,.588]); ни одного прежнего short-hit не потеряно при по-ID проверке.
Но это не общий успех локализации: FP tokens выросли, micro F1 упал.
У B7 среди верно detected positives peak приходится на gold в118/160 full против
120/190 pool. Верный response flag не гарантирует локализацию настоящей ошибки.

Ухудшение не ограничено длинными ответами. Предзаданные bins<=100,101–200,>200 tokens:
182/306/154 ответов. B7 full→sentence MF1: .6917→.5315, .7549→.5749,
.7698→.5982. Внутри bins иная prevalence, сравнивать эти MF1 с общим напрямую нельзя.
Полные class counts/CM, overlapping annotation-type slices и CI сохранены в JSON.

**Порядок evidence:** хотя набор информации тот же, решение меняется хотя бы при
одной из двух rotations у B7 52/642=8.10% (source CI6.07–10.28%), R6-B44/642=6.85%
(4.98–8.88%). Среднее |Δ max-risk|, усреднённое по двум rotations: B7 .04479
[.04003,.05021], R6-B .04745 [.04150,.05411]; среднее |Δ token probability| .01648/.01502.
Операционный criterion position-sensitivity выполнен. Rotations не дали убедительного
прироста MF1: все paired CI включают0. Не выбираем «удачную» перестановку как модель.
Это измеренная чувствительность, не доказательство единственного внутреннего механизма.
Явные междокументные ссылки по порядку могут ограничивать семантическую invariance.

**Что различил pilot:**
1. H1 о простом полезном снятии passage competition отвергнута для этого frozen
   pipeline: FPR растёт .169→.514 (B7), .151→.384 (R6-B), качество и localization
   падают; pipeline дороже. Обобщать это на специально обученные verifier нельзя.
2. H2 о потере совместного контекста поддерживается конкретными natural случаями.
   Одно грамматическое предложение может собирать несколько фактов из разных
   passages: один passage для whole sentence — слишком сильное ограничение.
   Однако заранее опасавшееся падение positive recall от token_min **не наблюдается**:
   recall выше full у обеих моделей. Не приписываем результат заранее ожидаемому FN-механизму.
3. У token_min AUROC/AP близки к full, особенно у R6-B (.87168/.78404 против
   .87291/.78298), но fixed-point MF1 хуже. Это совместимо с context-induced risk
   shift и несовместимостью прежней рабочей точки, а не только с потерей ranking.
   Не доказано, что причина исключительно calibration; thresholds после результата
   не подбирали. Sentence_pool дополнительно ухудшает ranking.
4. H3 — ограниченный перенос natural representations/supervision — остаётся
   возможным, но не идентифицирован отдельно от context distribution shift,
   joint-evidence semantics, annotation ambiguity и max-aggregation. R9 не даёт
   права сказать «B7 вообще не использует evidence» или «нужен GNN».

## Ошибки: конкретные проверяемые случаи

Posthoc biased audit: до4 максимальных |Δrisk| в каждом направлении/label, без
изменения inference по IDs. Полные texts, annotations, offsets и passage choices
остались в ignored predictions/phase2b_r9_final/*audit_examples.json.
Это анализ corpus faithfulness, не рекомендации по темам исходных документов.

- **Новый FP12559/source14411:** supported длинный перечень объединяет сведения из
  всех трёх passages в одном предложении. B7 .01458→.97418 sentence_pool;
  token_min .48512 тоже выше frozen threshold. Отдельный passage не содержит полного
  перечня. Это конкретный контрпример «одно предложение — один достаточный passage».
- **Исправленный FN12503/source14401:** короткий заголовок «Mild case» противоречит
  указанному в source условию; .01888→.99520, gold span[91,100), новый short gold-hit.
  Более высокая чувствительность реальна, но не окупает массовых ложных тревог.
- **Исправленный FP12055/source14325:** .87908→.04640; answer соответствует одному
  passage, другие дают отличающийся procedural context. Локальный выбор иногда
  помогает; это не систематический выигрыш на естественных документах.
- **FP15150/source15446:** supported ответ собирает описание figs из трёх passages.
  Full .00964, исправленный sentence_pool .51537, token_min .49702. Даже некоторые
  прямо поддержанные отдельным passage предложения получают повышенный риск.
  Это также совместимо с изменением распределения контекста у frozen token head;
  только недостатком multi-passage support все ошибки объяснить нельзя.

B7:34 исправленных responses (4FP+30FN),155 новых FP,0 новых FN.
R6-B:44 исправленных (7FP+37FN),109 новых FP,0 новых FN. Это counts ответов,
не независимых источников. Категории выше не оценены как population prevalence.

## Ошибки реализации и прозрачность

1. Первый full-B7 replay остановился на Python tuple vs JSON list в offsets.
   Six-record audit подтвердил exact probabilities; значения offsets тоже одинаковы.
   Исправлен только equality comparison, сохранены freeze/log и новый source hash.
2. После v2 case audit обнаружена неверная привязка BPE leading-space токенов к
   предыдущему предложению. Поэтому исходная sentence_pool статистика отозвана.
3. Строгий v3 crossing guard остановился на одном punctuation-only токене с
   question mark/closing quote. Whole-snapshot CPU audit:1 такой token/1 ответ.
   Итоговая generic rule: anchor — первый alphanumeric character; если их нет,
   первый non-whitespace, иначе original start. Все alphanumeric chars должны
   принадлежать одному предложению. Никаких ID exceptions; lexical crossing raises.

Original offsets/token probabilities не менялись. Добавлены regression fixtures;
final pooled outputs пересчитаны **без нового GPU** из проверенных saved forwards.
Token_min сохранился полностью, raw/rotation/single predictions скопированы byte-for-byte.
Proposal/гипотезы/порог/данные/gates не изменились; source freeze версии1/2/3/4 сохранены.
Implementation repair был после case inspection: результаты тем более exploratory.
См. R9_EXECUTION_AMENDMENT.md, R9_ALIGNMENT_CORRECTION*.md. Для новых conclusions
использовать только reports/phase2b_r9_final, а не reports/phase2b_r9.

## Ресурсы, тесты, artifacts

Сначала longest-input smoke/profile обоих checkpoints. Прогноз верхней оценки
563.94s;12 успешных resident pipeline stages заняли305.76s суммарно. Это inference
с encoding; load/hash/disk/bootstrap/tests/failed replay не включены. Новых fits0.
FP32 peak allocated VRAM690960896 bytes≈.644GiB; peak RSS≈1.44GiB.

| Model | Full ms/answer | Full answers/s | Pool stage-sum ms/answer | Pool stage-sum answers/s |
|---|---:|---:|---:|---:|
| B7 | 41.89 | 23.87 | 113.69 | 8.80 |
| R6B | 41.17 | 24.29 | 113.40 | 8.82 |

Pool timing — сумма трёх сохранённых inference stages + отдельно измеренный CPU
pooling, не единый production latency benchmark; load/cache excluded, один timing trial.

- Final offline pytest: **234 passed,8 skipped (68.35s)**.
- Final research pytest: **241 passed,1 skipped (80.71s)**.
- 12 R9 offline regression cases; leading-space/quote/crossing/coverage/provenance tested.
- Ruff passed; pip check обеих venv: No broken requirements found; dependencies не менялись.
- GROUNDCHECK_RAGTRUTH_DIR unset, legacy outer-dev regression отключён.
- .github/workflows/ci.yml tracked, Python3.11/3.12 offline. Remote CI не запускался/не подтверждён.
- Оба full replay совпали с historical R6 token probabilities **точно на642/642**.
- Проверены111/85/91 manifest entries R6/R7/R8; старые snapshots, checkpoints и review pack сохранены.
- Snapshot research_snapshots/R9_source_20261025.zip + member hashes в соседнем JSON.
  Local manifest models/phase2b_r9/R9_artifact_manifest.json включает исходные и
  исправленные версии. Text-bearing predictions/weights ignored; source ZIP их не содержит.
  Ничего не committed/pushed. Review R8 остаётся unreviewed; human gold не появился.

Главные изменённые/новые файлы: research/passage_probe.py, scripts/research_r9.py,
tests/test_passage_probe.py, docs/R9_RESEARCH_PROPOSAL.md, этот отчёт, три execution/alignment
amendments, PROGRESS.md, DECISIONS.md, EVALUATION_PROTOCOL.md, RESEARCH_NOTES.md.
Локальные diagnostics_final.py/report.py/seal.py входят в source snapshot, несмотря
на ignored models/. Checkpoint SHA: B7 7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4;
R6-B 86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e.

## Следующий наиболее информативный опыт — ещё не запущен

**Natural full-evidence order augmentation, CE-only.** Сохранять весь контекст,
не вводить extractor и не учить на очередных numeric templates. Matched control
raw+raw против raw+deterministic cyclic passage permutation на том же inner-fit,
одинаковый B7 start, seed42, optimizer, supervised token targets/exposures/steps.
Без ranking/consistency loss, нового aggregation и threshold search. Сначала
проверить admissibility перестановок (явные межpassage order references требуют
отдельного правила), profile; на evaluation применять заранее все rotations,
не выбирать лучшую. Hypothesis: label-preserving natural training снижает
измеренную position sensitivity без потери исходного natural/localization качества.

Заранее предлагаемый gate следующего пакета: relative reduction>=25% доли responses
с rotation flips, paired-source interval для изменения ниже0; original-order
macro-F1 noninferiority margin .01 с нижней границей CI>−.01; token-F1 не ниже и
никаких потерянных short gold-hit. Это проект протокола, его нужно заморозить до fit.
Ресурсная оценка по R6: до двух примерно10-minute fits (около20GPU-min, один seed);
это экстраполяция, а не измеренный R9 training budget. Повторные seeds только после gate.
Existing inner используется как development; для confirmation потребуется заранее
согласованный untouched evaluation и внешняя задача с проверкой происхождения,
а official test остаётся закрытым. R9 не разрешает автоматически запускать это обучение.

Почему не просто поднять threshold pooling: сохранённый ranking делает это разумной
будущей **train-side** calibration гипотезой, но posthoc подбор сейчас скрыл бы
провал frozen procedure. И даже удачный threshold не доказал бы более точный grounding.
Почему не автоматически dependency parser/GNN: R8 не активировал structural verification;
его нулевое покрытие не устанавливает оптимальную архитектуру. Измеренная natural
нестабильность полного verifier даёт более дешёвую проверяемую следующую гипотезу.
Это известный augmentation-приём, не заявление научной новизны.

## Воспроизведение

Не перезаписывать завершённые exclusive outputs. В текущей копии — tests и read-only
hash verification. Для replay восстановить source ZIP в **отдельную копию**, сохранить
old R6/R7/R8 manifests/archives и исходные v2 predictions/runtime/manifest R9; исключить
только final outputs, freeze_v4.json и новый report diagnostics. Веса/data локальные,
не включены в ZIP. Никакие guards старых freezes не обходить.

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\scripts"
$py='.venv-research\Scripts\python.exe'
Remove-Item Env:GROUNDCHECK_RAGTRUTH_DIR -ErrorAction SilentlyContinue
& $py scripts/research_r9.py verify-old
# В отдельной replay-копии, из сохранённых verified neural forwards (GPU не нужен):
& $py scripts/research_r9.py freeze
& $py scripts/research_r9.py repool
& $py scripts/research_r9.py analyze
& $py models/phase2b_r9/diagnostics_final.py
# Проверки в текущей копии (выберите свежие basetemp paths):
Remove-Item Env:GROUNDCHECK_RESEARCH_MODELS -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=models/phase2b_r9/repro_offline
$env:GROUNDCHECK_RESEARCH_MODELS='1'
& $py -m pytest -q -p no:cacheprovider --basetemp=models/phase2b_r9/repro_research
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pip check
& $py -m pip check
```

Для точного replay исходных GPU forwards использовать content-addressed source ZIP
из freeze_v2.json (первоначальная fixed equality версия) и его исходный proposal,
данные/checkpoints/lock; команды freeze, profile, run --model B7, run --model R6B.
Исправленная версия также умеет run для новой чистой копии, но сохранённый GPU
результат не нужно вычислять повторно ради исправления CPU aggregation.
Все законченные runs сохранены; незавершённого обучения и команды resume training нет.
