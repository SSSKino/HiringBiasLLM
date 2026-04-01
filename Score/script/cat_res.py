import argparse
import json
from glob import glob
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


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
        "chinese": "Chinese",
        "indian": "Indian",
        "africa": "Africa",
        "african": "Africa",
        "european": "European",
        "middle eastern": "Middle Eastern",
        "middleeastern": "Middle Eastern",
        "american": "American",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if not cleaned:
        return "NA"
    return cleaned.title()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize scores by race within each industry + seniority."
    )
    parser.add_argument(
        "--input",
        default=r".\cv_scores_exp1.json",
        help="Path to scores JSON or a directory of per-experiment JSON files",
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

    races_fixed = ["Chinese", "Indian", "Africa", "European", "Middle Eastern", "American"]

    for exp_name, scores in scores_by_exp.items():
        grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
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
            score = r.get("score_total")
            if isinstance(score, (int, float)):
                grouped[(industry, seniority)][race].append(float(score))

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        for (industry, seniority), race_scores in grouped.items():
            values = [agg_fn(race_scores[r]) if race_scores.get(r) else 0 for r in races_fixed]

            plt.figure(figsize=(8, 4.5))
            x = list(range(len(races_fixed)))
            plt.bar(x, values)
            plt.xticks(x, races_fixed, rotation=0)
            plt.ylim(0, 100)
            plt.title(f"{industry} / {seniority} - {args.stat} score by race")
            plt.xlabel("Race (name_category)")
            plt.ylabel("Score (0-100)")
            plt.tight_layout()

            safe_industry = industry.replace(" ", "_")
            safe_seniority = seniority.replace(" ", "_")
            out_path = exp_dir / f"{safe_industry}_{safe_seniority}_{args.stat}.png"
            plt.savefig(out_path)
            if args.show:
                plt.show()
            plt.close()
            total_charts += 1

    print(f"Wrote {total_charts} chart(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
