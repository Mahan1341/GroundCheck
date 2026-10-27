"""Build the bounded-loop verdict from saved measurements, without model execution."""

import re

from grounded_hallucination_lab.research.protocol import ROOT, read_json, write_json

BASE = ROOT / "models/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"


def ci(value):
    return f"[{value['low']:.4f}, {value['high']:.4f}]"


def prose_spacing(text):
    text = re.sub(r"(?<=[А-Яа-яЁё])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[А-Яа-яЁё])", " ", text)
    words = (
        "sources|source|negative|positive|parents|records|responses|examples|tokens|"
        "words|word|claims|claim|clusters|artifacts|seed|samples|test|train|human|"
        "historical|discard|coefficient|inner|reference|CUDA|GPU|MiB|GiB|bytes|min|"
        "pytest|passed|skipped|fit|checkpoints|hypotheses|training|MuSiQue|wordwindows|FactCG|matrix"
    )
    text = re.sub(rf"\b({words})(?=\d)", r"\1 ", text)
    return re.sub(rf"(?<=\d)(?=(?:{words})\b)", " ", text)


def main():
    c2, c4 = [read_json(REPORT / f"cycle{i}_analysis.json") for i in (2, 4)]
    external = {
        p: read_json(REPORT / f"external_{p}_three_analysis.json")
        for p in ("development", "confirmation")
    }
    jobs = []
    for path in sorted(REPORT.glob("*_resources.json")):
        r = read_json(path)
        seconds = r.get("wall_seconds_with_loading", r.get("seconds", r.get("pipeline_seconds")))
        if "name" in r:
            freeze = BASE / f"{r['name']}_run_freeze.json"
            if freeze.exists():
                seconds = max(seconds, path.stat().st_mtime - freeze.stat().st_mtime)
        jobs.append(
            {
                "job": path.stem,
                "charged_wall_seconds": seconds,
                "peak_allocated_bytes": r.get("cuda_peak_allocated_bytes"),
            }
        )
    profile_seconds = (BASE / "cycle3_profile/result.json").stat().st_mtime - (
        BASE / "cycle3_freeze.json"
    ).stat().st_mtime
    tests = read_json(REPORT / "research_tests_runtime.json")
    jobs += [
        {"job": "cycle3_gradient_pilot", "charged_wall_seconds": profile_seconds},
        {"job": "research_tests_entire_wall", "charged_wall_seconds": tests["seconds"]},
    ]
    minutes = sum(j["charged_wall_seconds"] for j in jobs) / 60
    assert minutes < 120 and tests["exit_code"] == 0
    confirmation = external["confirmation"]["all"]
    b, f = confirmation["metrics"]["B7"], confirmation["metrics"]["FactCG"]
    external_gate = {
        "gain_ge_03": f["macro_f1"] - b["macro_f1"] >= 0.03,
        "lower_ci_positive": confirmation["bootstrap"]["difference_from_reference"]["FactCG"][
            "macro_f1"
        ]["low"]
        > 0,
        "fpr_retained": f["fpr"] <= b["fpr"] + 0.05,
        "recall_retained": f["positive_class"]["recall"] >= b["positive_class"]["recall"] - 0.05,
    }
    write_json(
        REPORT / "resource_ledger.json",
        {
            "jobs": jobs,
            "total_charged_gpu_minutes": minutes,
            "cap_minutes": 120,
            "measurement": "Conservative GPU-process wall incl load/tokenization; "
            "all test wall charged; "
            "cycle3 freeze-to-result wall. Not hardware GPU-active time. No overlapping GPU jobs.",
            "training_fits": 0,
            "new_checkpoints": 0,
            "external_gate": external_gate,
            "external_gate_passed": all(external_gate.values()),
        },
    )
    lines = [
        "# GroundCheck — verdict bounded discovery loop, 2026-10-26",
        "",
        "Четыре цикла завершены; исследование продолжается, release не готовился. "
        "Рассмотрены8 гипотез из4 семейств. Проверены4 основные гипотезы, дополнительно "
        "выполнены annotation/provenance audit и однократная внешняя confirmation.",
        "",
        "**Согласованного улучшения natural detection, factual sensitivity и localization нет.** "
        "B7 остаётся лучшим из этих методов на reused RAGTruth inner. FactCG существенно "
        "сильнее на factual interventions и переносится лучше на внешнюю QA-задачу. "
        "Это результат готовой модели и нового сравнения, не собственная новая архитектура.",
        "",
        "## Что реально проверено",
        "",
        "| Цикл | Проверка | Результат и решение |",
        "|---|---|---|",
        "|1/H1|Max только по alphanumeric tokens, те же probabilities|"
        "68 natural scores изменились; "
        "решений0, ΔMF1 CI[0,0]. Гипотеза существенного улучшения отвергнута.|",
        "|2/H2|Pinned FactCG, author evidence packing + extractive claims|Контролируемые связи "
        "различает сильно; natural FP и coarse localization провалили replacement gate.|",
        "|3/H5|96 training-only FP32 forwards: actual logit-gradient mass|Для32 коротких ошибок "
        "медианная positive share=.4002, не<.20. Gate закрыт; "
        "backward/profile/fit не запускались.|",
        "|4/H3|Тот же FactCG, весь answer вместо clean fragments|FP меньше, FN значительно больше; "
        "короткие ошибки18→13/19. Gate провален; никаких пороговых исправлений.|",
        "",
        "Cycle3 treatment увеличил бы logit-gradient share до.8764, сохранив negative coefficients "
        "и суммарный positive coefficient 558.7074; "
        "это расчёт на frozen logits, не результат обучения. "
        "Малое число error tokens не означает малый gradient: трудные positive tokens дают "
        "существенный сигнал. Encoder-gradient interference этим не проверена.",
        "",
        "## Natural development: одинаковые642 ответа /107sources",
        "",
        "438 negative / 204 positive; include/all. Это многократно использованный inner, "
        "не независимое "
        "подтверждение. Состав R9/R10 сохранён: source15181 был исключён ранее по shared-passage "
        "правилу; historical648 и outer partitions не менялись.",
        "",
        "|Метод|Accuracy|Macro-F1 [95%CI]|Positive P/R/F1|AUROC|AP|FPR/FNR|Token F1|",
        "|---|---:|---|---|---:|---:|---|---:|",
    ]
    for name, source, key in (
        ("B7", c2, "B7"),
        ("FactCG segmented", c2, "FactCG"),
        ("FactCG whole", c4, "FactCG"),
    ):
        r = source["natural"]
        m, p = r["metrics"][key], r["metrics"][key]["positive_class"]
        lines.append(
            f"|{name}|{m['accuracy']:.4f}|{m['macro_f1']:.4f} "
            f"{ci(r['bootstrap']['absolute'][key]['macro_f1'])}|"
            f"{p['precision']:.4f}/{p['recall']:.4f}/{p['f1']:.4f}|"
            f"{m['auroc']:.4f}|{m['average_precision']:.4f}|"
            f"{m['fpr']:.4f}/{m['fnr']:.4f}|{r['localization'][key]['token']['f1']:.4f}|"
        )
    lines += [
        "",
        "Confusion matrices (true rows0/1): B7 [[364,74],[44,160]]; segmented "
        "[[231,207],[11,193]]; whole [[309,129],[87,117]]. Short TP / gold-token-hit "
        "из19: B7 14/8, segmented18/14, whole13/13. Whole помечает весь answer, поэтому "
        "его hit не является точной локализацией. Character F1 и exact-span F1, denominators "
        "и все raw counts сохранены в cycle2_analysis.json/cycle4_analysis.json. "
        "Scorer собственный: token any gold-overlap, nonwhite characters, exact trimmed merged "
        "spans приtoken>=.5; не официальный RAGTruth scorer.",
        "",
        "Segmented−B7 ΔMF1=−.13631,95%[−.18699,−.09032]. Whole−segmented ΔMF1=−.02874 "
        "[−.08083,.02741]; ΔFPR=−.17808[−.24243,−.11899], ΔFNR=+.37255[.29523,.45002]. "
        "Объединение shared-passage sources14325/15545 в106clusters даёт ΔMF1 "
        "[−.08302,.02833], без смены вывода. Bootstrap2000/seed42 парный поsource, "
        "не устраняет adaptive research bias и не заменяет training seeds.",
        "",
        "## Factual interventions: synthetic, отдельная постановка",
        "",
        "|Suite/метод|N/parents|Macro-F1|Both-correct pair accuracy|Signed claim/slot gap|",
        "|---|---|---:|---:|---:|",
    ]
    for label in ("r1", "r6"):
        for name, src, key in (
            ("B7", c2, "B7"),
            ("segmented", c2, "FactCG"),
            ("whole", c4, "FactCG"),
        ):
            r = src[label]
            q = r["paired"]["models"][key]
            lines.append(
                f"|{label}/{name}|{r['n']}/{r['sources']}|"
                f"{r['metrics'][key]['macro_f1']:.4f}|{q['pair_accuracy']:.4f}|"
                f"{q['local_gap']:.4f}|"
            )
    lines += [
        "",
        "Same-answer evidence worlds: это чувствительность к конкретным отношениям, "
        "не blank-evidence proxy. Все варианты одного parent остаются одним bootstrap cluster. "
        "R6 segmented−B7 local-gap CI[.47902,.53340]. Whole−segmented R1 pair Δ=−.05333 "
        "[−.10667,−.01333]. Успех здесь не переносится автоматически в natural.",
        "",
        "Safe R10 order subset204/34: any-rotation flips B7=18, segmented=24, whole=21; "
        "mean absolute risk Δ=.04204/.04870/.04344. Улучшение semantic discrimination "
        "не сопровождалось большей invariance. "
        "Ни лучший rotation, ни новый threshold не выбирались.",
        "",
        "## Внешняя human-triplet-derived оценка",
        "",
        "RefChecker/Dolly source split40dev/60locked заморожен до labels. До confirmation "
        "исключены6 источников одинаково для всех:4070 за training-text overlap FactCG и5 "
        "за complete-evidence capacity B7. Dev274/40 (82positive); confirmation363/54 "
        "(104positive/259negative).15 empty-triplet answers excluded as unknown вconfirmation. "
        "Human labels относятся к Claude2-extracted triplets; "
        "response label=any Neutral/Contradiction. "
        "Это не exhaustive response gold, никаких external span labels не выдумывали.",
        "",
        "|Раздел/метод|N/sources|Macro-F1|Positive F1|FPR|Recall+|ΔMF1 vsB7 95%CI|",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for partition, subset in (
        ("development", "all"),
        ("confirmation", "all"),
        ("confirmation", "closed_qa"),
        ("confirmation", "information_extraction"),
        ("confirmation", "summarization"),
    ):
        r = external[partition][subset]
        for name, m in r["metrics"].items():
            interval = (
                "reference"
                if name == "B7"
                else ci(r["bootstrap"]["difference_from_reference"][name]["macro_f1"])
            )
            lines.append(
                f"|{partition}/{subset}/{name}|{r['n']}/{r['sources']}|{m['macro_f1']:.4f}|"
                f"{m['positive_class']['f1']:.4f}|{m['fpr']:.4f}|"
                f"{m['positive_class']['recall']:.4f}|{interval}|"
            )
    lines += [
        "",
        "Preregistered primary external gate "
        f"**{'PASS' if all(external_gate.values()) else 'FAIL'}**. "
        "FactCG segmented−B7 ΔMF1=+.06991[.00302,.14408], FPR одинаковый66/259, "
        "TP74 вместо56/104. Whole выше по point estimate, но это заранее объявленная secondary "
        "ablation; не выбираем его победителем после confirmation. ClosedQA secondary:273/40, "
        "FactCG .71372 vsB7 .59322. Information extraction ухудшается:59ответов/9sources; "
        "summarization31/5 слишком мала для надёжного общего вывода. Category comparisons "
        "без multiplicity correction. Теперь confirmation использована: для следующего нового "
        "утверждения нужна новая untouched оценка.",
        "",
        "## Ошибки, confounds и annotation quality",
        "",
        "В SHA-selected12/147 новых FP (11sources), provisional assistant audit:4meta/absence/"
        "passage-reference,2courtesy/advice,3instructions/causal paraphrases,3possible annotation "
        "omissions. Это не independent human review и не оценка noise rate. Например, в12563 "
        "высокий риск у courtesy intro;16836 — causal safety paraphrase с evidence;16366 "
        "меняет P45 на Tax Return Copy, хотя gold0. Все gold сохранены. Whole подавляет6/12 "
        "тревог, но не подавляет3suspected omissions. Lost short TP:11937,11961,15663,15829,17596.",
        "",
        "Важный post-run confound: whole возвращает35 pure meta/refusal answers, которые "
        "clean_segments полностью исключал; все35 становятся FP. Поэтому C4 меняет также "
        "checkworthiness filtering, и не изолирует только discourse. Это обнаружено и "
        "задокументировано, без исправления и повторного подбора после результата. Только6natural "
        "ответов/1source имеют2evidence chunks: общие выводы о distributed support невозможны.",
        "",
        "17 mixed-span repeated-sentence groups/12sources не равны17label errors: heading "
        "и passage attribution меняют смысл. Automatic span propagation отвергнута. "
        "Annotation quality и detector quality нельзя объединять в одну метрику.",
        "",
        "## Стоимость и проверки",
        "",
        f"Conservative charged GPU-process wall **{minutes:.2f}/120min**, включая все research "
        "tests и model-loading intervals. Новых fit/checkpoints0; скачан один435M FactCG "
        "(1,740,304,440bytes), а не обучена новая модель. Все FactCG inputs<=908tokens "
        "в primary runs; confirmation<=820. Zero truncated pairs. Пиковая measured allocated "
        "VRAM FactCG2.02GiB; per-job latency/throughput/RSS/VRAM в *_resources.json. "
        "Cache reused exact evidence/claim pairs; timings нельзя выдавать за отдельный "
        "uncached benchmark каждого метода. Энергия/CUDA-active time не измерялись.",
        "",
        "Offline pytest282passed/8skipped; research opt-in289passed/1skipped. Последнийskip — "
        "реальные outer-data regression не разрешены. Ruff проходит с одной явной E501 "
        "waiver для102-character metadata line в immutable confirmation helper; helper/hash "
        "не менялись послеfreeze. Оба pip check: no broken requirements. CI файл tracked, "
        "matrix3.11/3.12 offline; remote CI не запускался/не подтверждён. Historical verification: "
        "R9 136artifacts,R10 120artifacts,B7/R10A/R10B checkpoints и oldsource guards PASS.",
        "",
        "## Происхождение, novelty и следующий шаг",
        "",
        "FactCG HF 0430e3509dbd28d2dff7a117c0eae25359ff3e80; "
        "author repo 41f185413312ed9b3904ec03963fe899693498e0, "
        "MIT, standard Transformers без custom remote code. Нет известного declared RAGTruth "
        "supervised overlap; доступные6training files не совпали16wordwindows с allowed R10 "
        "train/inner. Но5391MuSiQue premises отсутствуют в release, backbone pretraining "
        "непроверяемо. RefChecker source4070 исключён доlabels; это ограниченный audit, "
        "не гарантия отсутствия contamination. Primary literature и8hypotheses — "
        "RESEARCH_DISCOVERY_LOOP.md; все gates — R11_PLUS_RESEARCH_LOG.md и отдельные preregs.",
        "",
        "**Прорыва и доказанной научной новизны нет.** Есть честно подтверждённый external "
        "transfer gain готового verifier и опровержение двух простых remedies. Нельзя объявлять "
        "generic factual relation reasoning недоступным маленьким моделям: FactCG уже решает "
        "эти контрфактические пары. Нельзя обещать, что ещё один synthetic template или CE loss "
        "исправит natural detection: наши данные этого не поддерживают.",
        "",
        "Следующая фундаментальная задача — согласовать **что является проверяемым "
        "утверждением**, evidence-conditioned verdict и точную локализацию. Наиболее "
        "информативный следующий пакет: на новых train-side natural источниках отдельно "
        "оценить assertion/refusal/absence-policy selection, затем фиксированный FactCG verifier "
        "и B7 localization без изменения порогов. Gold factuality и checkworthiness нужны "
        "независимо, а не автоматически из нашего extractor. Три последующих проверки: "
        "(1) source-disjoint public human claim labels с контролем полноты extraction; "
        "(2) matched factual/nonfactual scope controls без выбрасывания фактов внутри отказа; "
        "(3) отдельный evidence-conditioned span objective только если первые две проверки "
        "покажут конкретный supervised signal. Эти опыты здесь не запускались.",
        "",
        "Команды, hashes, snapshot и полный перечень файлов: DISCOVERY_REPRODUCE.md. "
        "RAGTruth outer calibration/dev/test не открывались; публикации и release нет.",
        "",
    ]
    target = ROOT / "docs/RESEARCH_LOOP_VERDICT.md"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(prose_spacing("\n".join(lines)))
    print({"report": str(target), "charged_minutes": minutes, "external_gate": external_gate})


if __name__ == "__main__":
    main()
