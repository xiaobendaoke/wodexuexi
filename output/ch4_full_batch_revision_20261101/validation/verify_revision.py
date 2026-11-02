"""One-off manuscript checks and PDF previews; no simulation or new inference."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BEFORE = OUT.parent / "before"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def chapter_starts(doc):
    return {
        b.get_title(): b.get_dest().get_index() + 1
        for b in doc.get_toc()
        if b.level == 0 and b.get_dest()
    }


old = (BEFORE / "latex/docs/chap04.tex").read_text(encoding="utf-8-sig")
new = (ROOT / "latex/docs/chap04.tex").read_text(encoding="utf-8-sig")
hashes = read_json(OUT / "evidence_hashes_before.json")
unchanged = all(
    hashlib.sha256(Path(row["Path"]).read_bytes()).hexdigest().lower()
    == row["Hash"].lower()
    for row in hashes
)
source_files = [ROOT / "latex/docs" / n for n in (
    "chap04.tex", "abstract.tex", "chap01.tex", "chap06.tex"
)]
texts = [p.read_text(encoding="utf-8-sig") for p in source_files]
all_tex = "\n".join(p.read_text(encoding="utf-8-sig") for p in (ROOT / "latex").rglob("*.tex"))
labels = re.findall(r"\\label\{([^}]+)\}", all_tex)
refs = re.findall(r"\\(?:eqref|ref|autoref)\{([^}]+)\}", new)
unresolved = sorted(set(refs) - set(labels))
figures = lambda s: re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", s)

suite = read_json(ROOT / "results/canonical_all_20260920/chapter4/suite.json")
analysis = read_json(ROOT / "results/canonical_all_20260920/chapter4/analysis.json")
bridge = read_json(ROOT / "results/bridge_20260928/full/analysis.json")
units = analysis["units"]
main = {
    m: [u for u in units if u["variant"] == "main" and u["method"] == m]
    for m in ("mc_itr", "iterative_base")
}
main_means = {m: [mean(u["mean"][i] for u in us) for i in (0, 2, 4, 6)] for m, us in main.items()}
bridge_means = {
    m: mean(u[m]["deadline_misses"] for u in bridge["units"])
    for m in ("online_lower", "mc_itr")
}
pair = next(c for c in analysis["comparisons"] if c["variant"] == "main" and c["reference"] == "iterative_base")
plan = suite["plan"]
snapshots_per_config = len(plan["workloads"]) * plan["episodes"] * len(plan["snapshot_steps"])
counts = {
    "workload_seeds": len(plan["workloads"]),
    "episodes_per_seed": plan["episodes"],
    "snapshots_per_episode": len(plan["snapshot_steps"]),
    "snapshot_instances": len(plan["variants"]) * snapshots_per_config,
    "method_snapshot_evaluations": sum(u["snapshots"] for u in units),
    "method_configuration_seed_units": len(units),
    "bridge_snapshot_instances": bridge["snapshots"],
}
checks = {
    "evidence_and_figure_hashes_unchanged": unchanged,
    "all_eight_figure_references_preserved": figures(old) == figures(new) and len(figures(new)) == 8,
    "abstract_local_counterexample_removed": "tab:ch4-local-counterexample" not in new,
    "no_missing_chapter4_label_targets": not unresolved,
    "chapter4_labels_unique": all(Counter(labels)[label] == 1 for label in re.findall(r"\\label\{([^}]+)\}", new)),
    "incorrect_numbers_and_claims_removed": not any(x in "\n".join(texts) for x in (
        "4100", "10.771", "8.311", "22.8\\%", "1700", "无 Bug", "上界标杆", "改善幅度最大的", "Bootstrapstrap"
    )),
    "archived_evaluation_counts": counts["snapshot_instances"] == 6000 and counts["method_snapshot_evaluations"] == 24600 and len(units) == 410,
    "archived_main_means_match": abs(main_means["mc_itr"][0] - 2.055) < 1e-12 and abs(main_means["iterative_base"][0] - 2.228333333333333) < 1e-12,
    "archived_bridge_means_match": abs(bridge_means["online_lower"] - 10.695) < 1e-12 and abs(bridge_means["mc_itr"] - 8.235) < 1e-12,
    "physical_calibration_boundary_present": "27\\,\\mathrm{kW}" in new and "公式代数自洽不能替代物理标定" in new,
    "search_evaluation_formula_present": "1+RH+r_{\\mathrm{acc}}" in new,
}
log = (ROOT / "latex/main.log").read_text(encoding="utf-8", errors="replace")
checks["no_undefined_references_citations_or_missing_glyphs"] = not re.search(
    r"(?:Reference|Citation).*undefined|There were undefined|multiply defined|Missing character:|^!", log, re.M
)
chapter4_log = log.split("(docs/chap04.tex", 1)[1].split("(docs/chap05.tex", 1)[0]
checks["no_overfull_boxes_in_chapter4"] = "Overfull" not in chapter4_log
checks["all_ten_main_seed_differences_negative"] = all(
    next(u["mean"][0] for u in main["mc_itr"] if u["workload_seed"] == base["workload_seed"])
    < base["mean"][0]
    for base in main["iterative_base"]
)


def subsection_lengths(text):
    result = {}
    for match in re.finditer(r"\\subsection\{([^\n]+)\}\s*\n\\label\{([^}]+)\}", text):
        tail = text[match.end():]
        end = re.search(r"\\(?:subsection|section)\{", tail)
        if match[2] in {
            "sec:ch4-main-results", "sec:ch4-sensitivity", "sec:ch4-cost-results",
            "sec:exact-small", "sec:ch4-bridge"
        }:
            result[match[2]] = {"title": match[1], "source_characters": len(tail[:end.start()] if end else tail)}
    return result

doc = pdfium.PdfDocument(ROOT / "latex/main.pdf")
before_doc = pdfium.PdfDocument(BEFORE / "latex/main.pdf")
starts, old_starts = chapter_starts(doc), chapter_starts(before_doc)
ch4 = next(v for k, v in starts.items() if k.startswith("第4章"))
ch5 = next(v for k, v in starts.items() if k.startswith("第5章"))
ch6 = next(v for k, v in starts.items() if k.startswith("第6章"))
old_ch4 = next(v for k, v in old_starts.items() if k.startswith("第4章"))
old_ch5 = next(v for k, v in old_starts.items() if k.startswith("第5章"))
aux = (ROOT / "latex/main.aux").read_text(encoding="utf-8")
contribution_page = int(re.search(r"\\newlabel\{sec:contributions\}\{\{[^}]*\}\{(\d+)\}", aux)[1]) + 18
extra = sorted(set([5, 6, 7, 8, 9, contribution_page, contribution_page + 1, ch5, ch6, ch6 + 1, ch6 + 2, ch6 + 3]))
render_dir = OUT / "pages"
render_dir.mkdir(exist_ok=True)
inventory = []
for number in sorted(set(range(ch4, ch5)) | set(extra)):
    page = doc[number - 1]
    page.render(scale=1.4).to_pil().save(render_dir / f"page_{number:03d}.png")
    tp = page.get_textpage()
    text = tp.get_text_range()
    (render_dir / f"page_{number:03d}.txt").write_text(text, encoding="utf-8")
    inventory.append({"physical_page": number, "characters": len(text.strip())})
    tp.close()
    page.close()

for block, begin in enumerate(range(ch4, ch5, 6), 1):
    sheet = Image.new("RGB", (1200, 1180), "#ddd")
    draw = ImageDraw.Draw(sheet)
    for j, number in enumerate(range(begin, min(begin + 6, ch5))):
        im = Image.open(render_dir / f"page_{number:03d}.png").convert("RGB")
        im.thumbnail((390, 550))
        x, y = (j % 3) * 400, (j // 3) * 590
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + 555), f"PDF {number}; thesis {number-18}", fill="black")
    sheet.save(OUT / f"chapter4_contact_{block}.jpg", quality=92)

result = {
    "checks": checks,
    "evidence_files_checked": len(hashes),
    "counts": counts,
    "archived_main_means_miss_energy_time_requests": main_means,
    "archived_main_miss_pair_ci95": [pair["ci95"][0][0], pair["ci95"][1][0]],
    "archived_bridge_miss_means": bridge_means,
    "before_pdf_pages": len(before_doc),
    "final_pdf_pages": len(doc),
    "before_chapter4_physical_pages": [old_ch4, old_ch5 - 1],
    "chapter4_physical_pages": [ch4, ch5 - 1],
    "chapter4_printed_pages": [ch4 - 18, ch5 - 19],
    "chapter4_source_characters_before_after": [len(old), len(new)],
    "result_subsections_before": subsection_lengths(old),
    "result_subsections_after": subsection_lengths(new),
    "final_pdf_sha256": hashlib.sha256((ROOT / "latex/main.pdf").read_bytes()).hexdigest(),
    "rendered_pages": inventory,
    "overfull_messages": re.findall(r"^Overfull[^\n]*", log, re.M),
    "unresolved_chapter4_targets": unresolved,
}
(OUT / "revision_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"checks": checks, "chapter4_pdf_pages": [ch4, ch5 - 1], "total_pdf_pages": len(doc)}, ensure_ascii=False, indent=2))
doc.close()
before_doc.close()
assert all(checks.values()), "At least one manuscript check failed"
