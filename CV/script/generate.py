import argparse
import json
import random
from pathlib import Path


def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"name file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"name file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc

    names: list[str] = []
    if isinstance(parsed, dict):
        for _, value in parsed.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        names.append(item.strip())
            elif isinstance(value, str) and value.strip():
                names.append(value.strip())
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
    else:
        raise SystemExit(f"name.json must be a dict or list, got {type(parsed).__name__}")

    if not names:
        raise SystemExit(f"no names parsed from: {path}")
    return names


def load_name_categories(path: Path) -> dict[str, list[str]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"name file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SystemExit("name.json must be a JSON object for per-category mode.")

    categories: dict[str, list[str]] = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            names = [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
        elif isinstance(value, str) and value.strip():
            names = [value.strip()]
        else:
            names = []
        if names:
            categories[str(key)] = names

    if not categories:
        raise SystemExit("no valid categories found in name.json")
    return categories


def load_cvs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"cv file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"cv.json must be a list, got {type(data).__name__}")
    return data


def load_names_with_categories(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(f"name file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"name file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc

    pairs: list[tuple[str, str]] = []
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        pairs.append((item.strip(), str(key)))
            elif isinstance(value, str) and value.strip():
                pairs.append((value.strip(), str(key)))
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                pairs.append((item.strip(), "NA"))
    else:
        raise SystemExit(f"name.json must be a dict or list, got {type(parsed).__name__}")

    if not pairs:
        raise SystemExit(f"no names parsed from: {path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CV copies with random names.")
    parser.add_argument("--cv", default="cv.json", help="Path to cv.json")
    parser.add_argument("--names", default="name.json", help="Path to name.json")
    parser.add_argument("--output", default="cv_random_names.json", help="Output JSON path")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="How many copies to generate per CV.",
    )
    parser.add_argument(
        "--category-count",
        type=int,
        default=None,
        help="If set, pick N categories and take 1 name from each.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cv_path = Path(args.cv)
    name_path = Path(args.names)
    output_path = Path(args.output)

    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    cvs = load_cvs(cv_path)

    output: list[dict] = []
    id_counter = 1
    if args.category_count is not None:
        categories = load_name_categories(name_path)
        category_keys = list(categories.keys())
        if args.category_count < 1:
            raise SystemExit("--category-count must be >= 1")
        if args.category_count > len(category_keys):
            raise SystemExit(
                f"--category-count ({args.category_count}) exceeds available categories ({len(category_keys)})."
            )
        category_keys = random.sample(category_keys, k=args.category_count)
        for item in cvs:
            if not isinstance(item, dict):
                continue
            for idx, key in enumerate(category_keys, start=1):
                name = random.choice(categories[key])
                clone = dict(item)
                if "name" in clone:
                    clone["name"] = name
                # set category before deriving candidate_id
                clone["name_category"] = key
                clone["region"] = key
                industry = str(clone.get("industry_target", "NA")).replace(" ", "_")
                seniority = str(clone.get("seniority_level", "NA")).replace(" ", "_")
                region = str(clone.get("name_category", "NA")).replace(" ", "_")
                clone["candidate_id"] = f"{industry}_{seniority}_{region}"
                clone["cv_id"] = item.get("candidate_id")
                output.append(clone)
    else:
        # default: if name.json is categorized, take one from each category
        if name_path.exists():
            raw = name_path.read_text(encoding="utf-8").strip()
        else:
            raw = ""
        use_category_mode = False
        if raw:
            try:
                parsed = json.loads(raw)
                use_category_mode = isinstance(parsed, dict)
            except json.JSONDecodeError:
                use_category_mode = False

        if use_category_mode:
            categories = load_name_categories(name_path)
            category_keys = list(categories.keys())
            for item in cvs:
                if not isinstance(item, dict):
                    continue
                for idx, key in enumerate(category_keys, start=1):
                    name = random.choice(categories[key])
                    clone = dict(item)
                    if "name" in clone:
                        clone["name"] = name
                    clone["name_category"] = key
                    clone["region"] = key
                    industry = str(clone.get("industry_target", "NA")).replace(" ", "_")
                    seniority = str(clone.get("seniority_level", "NA")).replace(" ", "_")
                    region = str(clone.get("name_category", "NA")).replace(" ", "_")
                    clone["candidate_id"] = f"{industry}_{seniority}_{region}"
                    clone["cv_id"] = item.get("candidate_id")
                    output.append(clone)
        else:
            name_pairs = load_names_with_categories(name_path)
            for item in cvs:
                if not isinstance(item, dict):
                    continue

                if args.n <= len(name_pairs):
                    chosen = random.sample(name_pairs, k=args.n)
                else:
                    chosen = [random.choice(name_pairs) for _ in range(args.n)]

                for idx, (name, category) in enumerate(chosen, start=1):
                    clone = dict(item)
                    if "name" in clone:
                        clone["name"] = name
                    clone["name_category"] = category
                    clone["region"] = category
                    industry = str(clone.get("industry_target", "NA")).replace(" ", "_")
                    seniority = str(clone.get("seniority_level", "NA")).replace(" ", "_")
                    region = str(clone.get("name_category", "NA")).replace(" ", "_")
                    clone["candidate_id"] = f"{industry}_{seniority}_{region}"
                    clone["cv_id"] = item.get("candidate_id")
                    output.append(clone)

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    implicit_output: list[dict] = []
    for item in output:
        clone = dict(item)
        region = str(clone.get("region", clone.get("name_category", "NA")))
        summary = str(clone.get("summary", "")).strip()
        appendix = f"Coming from {region}."
        if summary:
            if not summary.endswith((".", "!", "?")):
                summary += "."
            summary = f"{summary} {appendix}"
        else:
            summary = appendix
        clone["summary"] = summary
        implicit_output.append(clone)

    implicit_path = output_path.with_name("cv_random_name_implicit.json")
    implicit_path.write_text(
        json.dumps(implicit_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(output)} records to {output_path.resolve()}")
    print(f"Wrote {len(implicit_output)} records to {implicit_path.resolve()}")


if __name__ == "__main__":
    main()
