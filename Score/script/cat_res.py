import argparse
import json
from glob import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import textwrap

import matplotlib.pyplot as plt

# Ensure readable axis labels on Windows
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_scores(path: Path, experiment: str | None, score_glob: str) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    paths: list[Path] = []
    if path.is_dir():
        paths = [Path(p) for p in glob(str(path / score_glob))]
        if not paths:
            raise SystemExit(f"no json files found in directory: {path} with pattern {score_glob}")
    else:
        paths = [path]

    results_by_exp: dict[str, dict] = {}

    for p in paths:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        data = json.loads(text)
        if isinstance(data, dict) and "experiments" in data:
            for exp in data.get("experiments", []):
                if not isinstance(exp, dict):
                    continue
                exp_name = exp.get("name") or p.stem
                if experiment and exp_name != experiment:
                    continue
                exp_key = exp_name
                results_by_exp.setdefault(exp_key, {"scores": [], "sources": set()})
                results_by_exp[exp_key]["sources"].add(p.stem)
                for item in exp.get("results", []):
                    if isinstance(item, dict) and "results" in item:
                        for r in item.get("results", []):
                            if isinstance(r, dict):
                                results_by_exp[exp_key]["scores"].append(r)
                    elif isinstance(item, list):
                        for r in item:
                            if isinstance(r, dict):
                                results_by_exp[exp_key]["scores"].append(r)
                    elif isinstance(item, dict):
                        results_by_exp[exp_key]["scores"].append(item)
        elif isinstance(data, list):
            # legacy list format
            exp_key = p.stem
            results_by_exp.setdefault(exp_key, {"scores": [], "sources": set()})
            results_by_exp[exp_key]["sources"].add(p.stem)
            for item in data:
                if isinstance(item, dict) and "results" in item:
                    for r in item.get("results", []):
                        if isinstance(r, dict):
                            results_by_exp[exp_key]["scores"].append(r)
                elif isinstance(item, list):
                    for r in item:
                        if isinstance(r, dict):
                            results_by_exp[exp_key]["scores"].append(r)
                elif isinstance(item, dict):
                    results_by_exp[exp_key]["scores"].append(item)

    return results_by_exp


def parse_candidate_id(candidate_id: str) -> tuple[str, str, str]:
    parts = candidate_id.split("_")
    if len(parts) >= 3:
        race = "_".join(parts[2:]).replace("_", " ")
        return parts[0], parts[1], race
    if len(parts) == 2:
        return parts[0], parts[1], "NA"
    return "NA", "NA", "NA"


def normalize_race(value: str) -> str:
    cleaned = value.replace("_", " ")
    for h in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        cleaned = cleaned.replace(h, "-")
    cleaned = " ".join(cleaned.split()).strip().lower()
    mapping = {
        "sub-saharan africa": "Sub-Saharan Africa",
        "sub saharan africa": "Sub-Saharan Africa",
        "northern africa and western asia": "Northern Africa and Western Asia",
        "central and southern asia": "Central and Southern Asia",
        "central & southern asia": "Central and Southern Asia",
        "southern asia": "Central and Southern Asia",
        "southen asia": "Central and Southern Asia",
        "eastern and south-eastern asia": "Eastern and South-Eastern Asia",
        "eastern and south eastern asia": "Eastern and South-Eastern Asia",
        "latin america and the caribbean": "Latin America and the Caribbean",
        "australia and new zealand": "Oceania",
        "oceania": "Oceania",
        "europe and northern america": "Europe and Northern America",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if not cleaned:
        return "NA"
    return cleaned.title()


def normalize_record(r: dict) -> dict:
    # 适配多种格式：
    # 1) 新格式：顶层直接有 6 个维度 (Skill Match, Experience Match, etc.)
    # 2) 旧格式：scores.objective_score 和 scores.subjective_score
    
    # 新格式的 6 个维度名称映射
    new_format_dimensions = {
        "Skill Match": None,
        "Experience Match": None,
        "Education Match": None,
        "Communication and Collaboration": None,
        "Execution, Compliance, and Reliability": None,
        "Role-context Adaptability": None,
    }
    
    # 检查是否是新格式（直接包含维度）
    new_scores = {}
    for dim_name in new_format_dimensions.keys():
        if dim_name in r and isinstance(r[dim_name], dict):
            score = r[dim_name].get("score")
            if isinstance(score, (int, float)):
                new_scores[dim_name] = score
    
    if new_scores:
        # 新格式：计算 6 个维度的平均分
        obj_avg = sum(new_scores.values()) / len(new_scores) if new_scores else 0.0
        total_avg = obj_avg
    else:
        # 旧格式处理
        scores = r.get("scores", {})
        
        # 尝试从 scores.objective_score 获取
        if "objective_score" in scores:
            obj = scores.get("objective_score", {})
        # 或者从顶层 objective_score 获取
        elif "objective_score" in r:
            obj = r.get("objective_score", {})
        else:
            # fallback: objective scores provided at top-level
            obj = {k: v for k, v in scores.items() if k in ["skills_match", "experience_match", "education_match"]}
        
        # 只从 scores 中获取 subjective_score (如果存在)
        subj = scores.get("subjective_score", {})

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        obj_avg = avg([v for v in obj.values() if isinstance(v, (int, float))])
        subj_avg = avg([v for v in subj.values() if isinstance(v, (int, float))])
        # 如果没有 subjective_score，只用 objective_score 不求平均
        total_avg = obj_avg if subj_avg == 0.0 else (obj_avg + subj_avg) / 2.0

    normalized = dict(r)
    normalized["group_scores"] = {
        "objective_score_avg": round(obj_avg, 2),
        "total_avg": round(total_avg, 2),
    }
    normalized["score_total_normalized"] = round(total_avg, 2)
    return normalized




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize scores by race within each industry + seniority."
    )
    parser.add_argument(
        "--input",
        default=r".",
        help="Path to scores JSON or a directory of per-experiment JSON files",
    )
    parser.add_argument(
        "--score-glob",
        default="*score*.json",
        help="Glob pattern for score files when input is a directory.",
    )
    parser.add_argument(
        "--use-input-name",
        action="store_true",
        help="Prefix chart filenames with the input JSON base name.",
    )
    parser.add_argument("--output-dir", default=r".\charts", help="Output directory for charts")
    parser.add_argument(
        "--stat",
        default="mean",
        choices=["mean", "median"],
        help="Aggregation statistic for scores",
    )
    args = parser.parse_args()

    scores_by_exp = load_scores(Path(args.input), None, args.score_glob)
    if not scores_by_exp:
        raise SystemExit("No scores found in input.")

    agg_fn = mean if args.stat == "mean" else median

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_charts = 0

    races_fixed = [
        "Sub-Saharan Africa",
        "Northern Africa and Western Asia",
        "Central and Southern Asia",
        "Eastern and South-Eastern Asia",
        "Latin America and the Caribbean",
        "Oceania",
        "Europe and Northern America",
    ]
    races_fixed_wrapped = [textwrap.fill(r, width=14) for r in races_fixed]

    for exp_name, bundle in scores_by_exp.items():
        scores = bundle["scores"]
        sources = sorted(list(bundle.get("sources", [])))
        source_name = exp_name if len(sources) != 1 else sources[0]
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        normalized_records = []
        for r in scores:
            candidate_id = str(r.get("candidate_id", "NA"))
            industry_id, seniority, race = parse_candidate_id(candidate_id)
            race = normalize_race(race)
            industry = industry_id if industry_id and industry_id != "NA" else str(r.get("industry_target", "")).strip()
            if not industry:
                continue
            normalized = normalize_record(r)
            normalized_records.append(normalized)
            score = normalized.get("group_scores", {}).get("objective_score_avg")
            if isinstance(score, (int, float)):
                grouped[(industry, seniority)][race].append(float(score))

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "normalized_scores.json").write_text(
            json.dumps(normalized_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for (industry, seniority), race_scores in grouped.items():
            values = [agg_fn(race_scores[r]) if race_scores.get(r) else 0 for r in races_fixed]

            plt.figure(figsize=(12, 6.5))
            x = list(range(len(races_fixed)))
            bars = plt.bar(x, values)
            plt.bar_label(bars, fmt='%.2f', padding=3)
            plt.xticks(x, races_fixed_wrapped, rotation=0, fontsize=9)
            plt.ylim(0, 100)
            plt.title(f"{industry} / {seniority} - {args.stat} score by race")
            plt.xlabel("Race (name_category)")
            plt.ylabel("Score (0-100)")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.28)

            safe_industry = industry.replace(" ", "_")
            safe_seniority = seniority.replace(" ", "_")
            prefix = ""
            if args.use_input_name:
                prefix = f"{source_name}_"
            out_path = exp_dir / f"{prefix}{safe_industry}_{safe_seniority}_{args.stat}.png"
            plt.savefig(out_path)
            plt.close()
            total_charts += 1

    print(f"Wrote {total_charts} chart(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
