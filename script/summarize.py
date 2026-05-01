import argparse
import json
from collections import Counter
from pathlib import Path


def load_cvs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"cv file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"cv.json must be a list, got {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def print_counter(title: str, counter: Counter, top: int | None) -> None:
    print(title)
    items = counter.most_common(top) if top else counter.most_common()
    for key, count in items:
        print(f"  {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CV dataset.")
    parser.add_argument("--cv", default="cv.json", help="Path to cv.json")
    parser.add_argument("--top", type=int, default=None, help="Show top N entries per group")
    args = parser.parse_args()

    cvs = load_cvs(Path(args.cv))
    total = len(cvs)
    industry = Counter()
    seniority = Counter()
    category = Counter()
    industry_seniority = Counter()

    for item in cvs:
        industry[item.get("industry_target", "NA")] += 1
        seniority[item.get("seniority_level", "NA")] += 1
        industry_seniority[(item.get("industry_target", "NA"), item.get("seniority_level", "NA"))] += 1
        if "name_category" in item:
            category[item.get("name_category", "NA")] += 1

    print(f"Total CVs: {total}")
    print_counter("By industry_target:", industry, args.top)
    print_counter("By seniority_level:", seniority, args.top)
    if industry_seniority:
        print("By industry_target + seniority_level:")
        for (ind, sen), count in sorted(industry_seniority.items()):
            print(f"  {ind} / {sen}: {count}")
    if category:
        print_counter("By name_category:", category, args.top)


if __name__ == "__main__":
    main()
