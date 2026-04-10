import argparse
import json
from glob import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

# Ensure readable axis labels on Windows
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_scores(path: Path, experiment: str | None) -> dict[str, list[dict]]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    paths: list[Path] = []
    if path.is_dir():
        paths = [Path(p) for p in glob(str(path / "*.json"))]
        if not paths:
            raise SystemExit(f"no json files found in directory: {path}")
    else:
        paths = [path]

    results_by_exp: dict[str, list[dict]] = {}

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
                results_by_exp.setdefault(exp_key, [])
                for item in exp.get("results", []):
                    if isinstance(item, dict) and "results" in item:
                        for r in item.get("results", []):
                            if isinstance(r, dict):
                                results_by_exp[exp_key].append(r)
                    elif isinstance(item, dict):
                        results_by_exp[exp_key].append(item)
        elif isinstance(data, list):
            # legacy list format
            exp_key = p.stem
            results_by_exp.setdefault(exp_key, [])
            for item in data:
                if isinstance(item, dict) and "results" in item:
                    for r in item.get("results", []):
                        if isinstance(r, dict):
                            results_by_exp[exp_key].append(r)
                elif isinstance(item, dict):
                    results_by_exp[exp_key].append(item)

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
    cleaned = " ".join(value.replace("_", " ").split()).strip().lower()
    mapping = {
        "sub-saharan africa": "Sub-Saharan Africa",
        "sub saharan africa": "Sub-Saharan Africa",
        "northern africa and western asia": "Northern Africa and Western Asia",
        "central and southern asia": "Central and Southern Asia",
        "eastern and south-eastern asia": "Eastern and South-Eastern Asia",
        "eastern and south eastern asia": "Eastern and South-Eastern Asia",
        "latin america and the caribbean": "Latin America and the Caribbean",
        "australia and new zealand": "Australia and New Zealand",
        "europe and northern america": "Europe and Northern America",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if not cleaned:
        return "NA"
    return cleaned.title()


def normalize_record(r: dict) -> dict:
    scores = r.get("scores", {})
    obj = scores.get("objective_score", {})
    subj = scores.get("subjective_score", {})

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    obj_avg = avg([v for v in obj.values() if isinstance(v, (int, float))])
    subj_avg = avg([v for v in subj.values() if isinstance(v, (int, float))])
    total_avg = (obj_avg + subj_avg) / 2.0

    normalized = dict(r)
    normalized["group_scores"] = {
        "objective_score_avg": round(obj_avg, 2),
        "subjective_impression_avg": round(subj_avg, 2),
        "total_avg": round(total_avg, 2),
    }
    normalized["score_total_normalized"] = round(total_avg, 2)
    return normalized


def get_by_path(obj: dict, path: str) -> float | None:
    cur: object = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize scores by race within each industry + seniority."
    )
    parser.add_argument(
        "--input",
        default=r".\cv_scores_exp1.json",
        help="Path to scores JSON or a directory of per-experiment JSON files",
    )
    parser.add_argument(
        "--use-input-name",
        action="store_true",
        help="Prefix chart filenames with the input JSON base name.",
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help="Dot path for score value (e.g., scores.objective_score.skills_match). "
        "If omitted, uses objective_score average.",
    )
    parser.add_argument("--output-dir", default=r".\charts", help="Output directory for charts")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Filter by experiment name (e.g., exp2_mask_candidate_and_cv_implicit).",
    )
    parser.add_argument(
        "--allowed-industries",
        default="Law,IT,HR,Finance",
        help="Comma-separated industry whitelist. Leave empty to disable filtering.",
    )
    parser.add_argument("--industry", default=None, help="Filter by industry")
    parser.add_argument("--seniority", default=None, help="Filter by seniority")
    parser.add_argument(
        "--stat",
        default="mean",
        choices=["mean", "median"],
        help="Aggregation statistic for scores",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--normalized-output-name",
        default="normalized_scores.json",
        help="Filename to write normalized scores under each experiment folder.",
    )
    args = parser.parse_args()

    scores_by_exp = load_scores(Path(args.input), args.experiment)
    if not scores_by_exp:
        raise SystemExit("No scores found in input.")

    agg_fn = mean if args.stat == "mean" else median

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_charts = 0
    allowed = {
        s.strip()
        for s in str(args.allowed_industries).split(",")
        if s is not None and s.strip()
    }

    races_fixed = [
        "Sub-Saharan Africa",
        "Northern Africa and Western Asia",
        "Central and Southern Asia",
        "Eastern and South-Eastern Asia",
        "Latin America and the Caribbean",
        "Australia and New Zealand",
        "Europe and Northern America",
    ]
    races_fixed_wrapped = [r.replace(" and ", "\n") for r in races_fixed]

    for exp_name, scores in scores_by_exp.items():
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        normalized_records = []
        for r in scores:
            candidate_id = str(r.get("candidate_id", "NA"))
            industry_id, seniority, race = parse_candidate_id(candidate_id)
            race = normalize_race(race)
            industry = str(r.get("industry_target", "")).strip()
            if not industry:
                continue
            if allowed and industry not in allowed:
                continue
            if args.industry and industry != args.industry:
                continue
            if args.seniority and seniority != args.seniority:
                continue
            normalized = normalize_record(r)
            normalized_records.append(normalized)
            if args.score_path:
                score = get_by_path(normalized, args.score_path)
            else:
                score = normalized.get("group_scores", {}).get("objective_score_avg")
            if isinstance(score, (int, float)):
                grouped[(industry, seniority)][race].append(float(score))

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / args.normalized_output_name).write_text(
            json.dumps(normalized_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for (industry, seniority), race_scores in grouped.items():
            values = [agg_fn(race_scores[r]) if race_scores.get(r) else 0 for r in races_fixed]

            plt.figure(figsize=(12, 5.5))
            x = list(range(len(races_fixed)))
            plt.bar(x, values)
            plt.xticks(x, races_fixed_wrapped, rotation=0, fontsize=9)
            plt.ylim(0, 100)
            plt.title(f"{industry} / {seniority} - {args.stat} score by race")
            plt.xlabel("Race (name_category)")
            plt.ylabel("Score (0-100)")
            plt.tight_layout()

            safe_industry = industry.replace(" ", "_")
            safe_seniority = seniority.replace(" ", "_")
            prefix = ""
            if args.use_input_name and isinstance(args.input, str):
                base = Path(args.input).name
                if base.endswith(".json"):
                    base = base[:-5]
                prefix = f"{base}_"
            out_path = exp_dir / f"{prefix}{safe_industry}_{safe_seniority}_{args.stat}.png"
            plt.savefig(out_path)
            if args.show:
                plt.show()
            plt.close()
            total_charts += 1

    print(f"Wrote {total_charts} chart(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
