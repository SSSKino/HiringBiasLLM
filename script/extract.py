from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas/openpyxl. Install with `pip install pandas openpyxl`."
    ) from exc


FIELDS_COLLECTED = [
    "soc_code",
    "title",
    "description",
    "skills",
    "abilities",
    "knowledge",
    "tasks",
    "technology_skills",
    "education_requirements",
    "work_activities",
    "work_context",
]


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized_candidates = {_normalize_name(c) for c in candidates}
    for col in columns:
        if _normalize_name(col) in normalized_candidates:
            return col
    return None


def _first_sheet(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if not sheets:
        return pd.DataFrame()
    return next(iter(sheets.values()))


def _load_occ_base(source_dir: Path) -> pd.DataFrame:
    df = _first_sheet(source_dir / "Occupation Data.xlsx")
    if df.empty:
        return pd.DataFrame(columns=["soc_code", "title", "description"])

    cols = [str(c) for c in df.columns]
    soc_col = _find_column(cols, ["O*NET-SOC Code", "SOC Code", "ONET-SOC Code", "soc_code"])
    title_col = _find_column(cols, ["Title", "Occupation Title", "occupation_title"])
    desc_col = _find_column(cols, ["Description", "Occupation Description", "description"])

    if soc_col is None:
        return pd.DataFrame(columns=["soc_code", "title", "description"])

    out = pd.DataFrame()
    out["soc_code"] = df[soc_col].astype(str).str.strip()
    out["title"] = df[title_col].fillna("").astype(str).str.strip() if title_col else ""
    out["description"] = df[desc_col].fillna("").astype(str).str.strip() if desc_col else ""
    out = out[out["soc_code"] != ""].drop_duplicates(subset=["soc_code"])
    return out


def _filter_occupations(base_df: pd.DataFrame, queries: list[str] | None, max_count: int | None) -> pd.DataFrame:
    out = base_df.copy()
    if queries:
        q = [x.strip().lower() for x in queries if x.strip()]

        def _match_row(row: pd.Series) -> bool:
            soc = row["soc_code"].lower()
            title = row["title"].lower()
            for one in q:
                # Keep SOC matching permissive for codes.
                if one in soc:
                    return True
                # Title matching uses word boundary to avoid substring noise:
                # e.g. "it" should not match "sustainability".
                if re.search(rf"\b{re.escape(one)}\b", title):
                    return True
            return False

        out = out[
            out.apply(_match_row, axis=1)
        ]
    if max_count is not None and max_count >= 0:
        out = out.head(max_count)
    return out


def _choose_scale(df: pd.DataFrame, scale_col: str | None) -> pd.DataFrame:
    if scale_col is None or df.empty:
        return df
    scale_series = df[scale_col].fillna("").astype(str).str.upper()
    for preferred in ("IM", "LV"):
        if (scale_series == preferred).any():
            return df[scale_series == preferred]
    return df


def _limit_rows(df: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    if top_n is None:
        return df
    return df.head(top_n)


def _rating_table(
    source_dir: Path,
    filename: str,
    name_candidates: list[str],
    value_candidates: list[str] | None = None,
    top_n: int | None = None,
) -> dict[str, list[dict]]:
    path = source_dir / filename
    if not path.exists():
        return {}

    df = _first_sheet(path)
    if df.empty:
        return {}

    cols = [str(c) for c in df.columns]
    soc_col = _find_column(cols, ["O*NET-SOC Code", "SOC Code", "ONET-SOC Code", "soc_code"])
    name_col = _find_column(cols, name_candidates)
    value_col = _find_column(cols, value_candidates or ["Data Value", "Value", "score"])
    scale_col = _find_column(cols, ["Scale ID", "Scale", "Scale Name"])

    if soc_col is None or name_col is None:
        return {}

    work = df.copy()
    work[soc_col] = work[soc_col].astype(str).str.strip()
    work[name_col] = work[name_col].fillna("").astype(str).str.strip()
    work = work[(work[soc_col] != "") & (work[name_col] != "")]
    work = _choose_scale(work, scale_col)

    if value_col:
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col]).sort_values([soc_col, value_col], ascending=[True, False])
    else:
        work = work.sort_values([soc_col, name_col], ascending=[True, True])

    result: dict[str, list[dict]] = {}
    for soc, group in work.groupby(soc_col):
        group = _limit_rows(group.drop_duplicates(subset=[name_col]), top_n)
        if value_col:
            items = [{"name": row[name_col], "score": float(row[value_col])} for _, row in group.iterrows()]
        else:
            items = [{"name": row[name_col]} for _, row in group.iterrows()]
        result[str(soc)] = items
    return result


def _tasks_table(source_dir: Path, top_n: int | None = None) -> dict[str, list[str]]:
    path = source_dir / "Task Statements.xlsx"
    if not path.exists():
        return {}

    df = _first_sheet(path)
    if df.empty:
        return {}

    cols = [str(c) for c in df.columns]
    soc_col = _find_column(cols, ["O*NET-SOC Code", "SOC Code", "ONET-SOC Code", "soc_code"])
    task_col = _find_column(cols, ["Task", "Task Statement", "task"])
    if soc_col is None or task_col is None:
        return {}

    work = df.copy()
    work[soc_col] = work[soc_col].astype(str).str.strip()
    work[task_col] = work[task_col].fillna("").astype(str).str.strip()
    work = work[(work[soc_col] != "") & (work[task_col] != "")]
    work = work.drop_duplicates(subset=[soc_col, task_col])

    out: dict[str, list[str]] = {}
    for soc, group in work.groupby(soc_col):
        out[str(soc)] = _limit_rows(group, top_n)[task_col].tolist()
    return out


def _technology_table(source_dir: Path, top_n: int | None = None) -> dict[str, list[dict]]:
    path = source_dir / "Technology Skills.xlsx"
    if not path.exists():
        return {}

    df = _first_sheet(path)
    if df.empty:
        return {}

    cols = [str(c) for c in df.columns]
    soc_col = _find_column(cols, ["O*NET-SOC Code", "SOC Code", "ONET-SOC Code", "soc_code"])
    name_col = _find_column(
        cols,
        ["Example", "Technology Skill", "Commodity Title", "Hot Technology Example"],
    )
    hot_col = _find_column(cols, ["Hot Technology", "Hot", "hot_technology"])
    if soc_col is None or name_col is None:
        return {}

    work = df.copy()
    work[soc_col] = work[soc_col].astype(str).str.strip()
    work[name_col] = work[name_col].fillna("").astype(str).str.strip()
    work = work[(work[soc_col] != "") & (work[name_col] != "")]

    out: dict[str, list[dict]] = {}
    for soc, group in work.groupby(soc_col):
        group = _limit_rows(group.drop_duplicates(subset=[name_col]), top_n)
        if hot_col:
            items = [
                {"name": row[name_col], "hot_technology": str(row[hot_col]) if pd.notna(row[hot_col]) else "N"}
                for _, row in group.iterrows()
            ]
        else:
            items = [{"name": row[name_col], "hot_technology": "N"} for _, row in group.iterrows()]
        out[str(soc)] = items
    return out


def _education_table(source_dir: Path) -> dict[str, dict]:
    path = source_dir / "Education, Training, and Experience.xlsx"
    if not path.exists():
        return {}

    df = _first_sheet(path)
    if df.empty:
        return {}

    cols = [str(c) for c in df.columns]
    soc_col = _find_column(cols, ["O*NET-SOC Code", "SOC Code", "ONET-SOC Code", "soc_code"])
    name_col = _find_column(cols, ["Element Name", "Category", "Education Level", "Title"])
    value_col = _find_column(cols, ["Data Value", "Value", "Percentage"])
    scale_col = _find_column(cols, ["Scale ID", "Scale", "Scale Name"])
    if soc_col is None or name_col is None or value_col is None:
        return {}

    work = df.copy()
    work[soc_col] = work[soc_col].astype(str).str.strip()
    work[name_col] = work[name_col].fillna("").astype(str).str.strip()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[(work[soc_col] != "") & (work[name_col] != "")]
    work = work.dropna(subset=[value_col])
    work = _choose_scale(work, scale_col)
    work = work.sort_values([soc_col, value_col], ascending=[True, False])

    out: dict[str, dict] = {}
    for soc, group in work.groupby(soc_col):
        top = group.drop_duplicates(subset=[name_col]).head(8)
        out[str(soc)] = {str(r[name_col]): float(r[value_col]) for _, r in top.iterrows()}
    return out


def _build_output(
    selected_occ_df: pd.DataFrame,
    occupations_key: str,
    source_dir: Path,
    top_n: int | None,
) -> dict:
    skills = _rating_table(source_dir, "Skills.xlsx", ["Element Name", "Skill", "name"], top_n=top_n)
    abilities = _rating_table(source_dir, "Abilities.xlsx", ["Element Name", "Ability", "name"], top_n=top_n)
    knowledge = _rating_table(source_dir, "Knowledge.xlsx", ["Element Name", "Knowledge", "name"], top_n=top_n)
    work_activities = _rating_table(
        source_dir,
        "Work Activities.xlsx",
        ["Element Name", "Work Activity", "name"],
        top_n=top_n,
    )
    work_context = _rating_table(
        source_dir,
        "Work Context.xlsx",
        ["Element Name", "Work Context", "Context", "name"],
        top_n=top_n,
    )
    tasks = _tasks_table(source_dir, top_n=top_n)
    tech = _technology_table(source_dir, top_n=top_n)
    education = _education_table(source_dir)

    occupation_items = []
    for _, row in selected_occ_df.iterrows():
        soc = row["soc_code"]
        occupation_items.append(
            {
                "soc_code": soc,
                "title": row["title"],
                "description": row["description"],
                "skills": skills.get(soc, []),
                "abilities": abilities.get(soc, []),
                "knowledge": knowledge.get(soc, []),
                "tasks": tasks.get(soc, []),
                "technology_skills": tech.get(soc, []),
                "education_requirements": education.get(soc, {}),
                "work_activities": work_activities.get(soc, []),
                "work_context": work_context.get(soc, []),
            }
        )

    return {
        "metadata": {
            "source": "O*NET Online Database",
            "url": "https://www.onetcenter.org/database.html",
            "industries": [occupations_key],
            "fields_collected": FIELDS_COLLECTED,
        },
        "occupations": {
            occupations_key: occupation_items,
        },
    }


def _parse_industry_keywords(raw: str) -> dict[str, list[str]]:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise SystemExit(f"Invalid --industry-keywords format: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SystemExit("--industry-keywords must be a Python dict, e.g. {'law': ['lawyer']}.")

    result: dict[str, list[str]] = {}
    for industry, keywords in parsed.items():
        key = str(industry).strip()
        if not key:
            continue
        if isinstance(keywords, str):
            result[key] = [keywords]
        elif isinstance(keywords, list):
            result[key] = [str(k) for k in keywords]
        else:
            raise SystemExit(
                f"Invalid keywords for industry '{industry}': use string or list of strings."
            )
    return result


def _load_industry_keywords_file(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise SystemExit(f"industry keywords file not found: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{path} must be a JSON object, e.g. {{\"IT\": [\"software\"]}}")

    result: dict[str, list[str]] = {}
    for industry, keywords in parsed.items():
        key = str(industry).strip()
        if not key:
            continue
        if isinstance(keywords, str):
            result[key] = [keywords]
        elif isinstance(keywords, list):
            result[key] = [str(k) for k in keywords]
        else:
            raise SystemExit(
                f"Invalid keywords for industry '{industry}' in {path}: use string or list."
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export O*NET db_30_2_excel to target JSON schema.")
    parser.add_argument("--source-dir", default="db_30_2_excel", help="Path of db_30_2_excel.")
    parser.add_argument("--output", default="onet_job_dataset.filtered.json", help="Output JSON file.")
    parser.add_argument(
        "--occupations",
        default="",
        help="Comma-separated filters by SOC code or title keyword, e.g. '15-1252.00,nurse,data scientist'.",
    )
    parser.add_argument(
        "--industry-keywords",
        default="",
        help=(
            "Python dict string for multi-industry mode, "
            "e.g. \"{'law':['lawyer','attorney'],'finance':['finance','bank']}\""
        ),
    )
    parser.add_argument(
        "--industry-keywords-file",
        default="industry_keywords.json",
        help="Path to JSON file mapping industry -> keyword list.",
    )
    parser.add_argument("--max-occupations", type=int, default=30, help="Maximum number of occupations.")
    parser.add_argument("--group-name", default="Selected", help="Key under occupations, e.g. IT.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Top N items for skills/abilities/etc. Default: unlimited.",
    )
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    base = _load_occ_base(source_dir)
    if args.industry_keywords:
        industry_map = _parse_industry_keywords(args.industry_keywords)
    elif args.industry_keywords_file and Path(args.industry_keywords_file).exists():
        industry_map = _load_industry_keywords_file(Path(args.industry_keywords_file))
    else:
        queries = [x.strip() for x in args.occupations.split(",")] if args.occupations else []
        industry_map = {args.group_name: queries}

    occupations_out: dict[str, list[dict]] = {}
    count_report: dict[str, int] = {}
    for industry, keywords in industry_map.items():
        selected = _filter_occupations(base, keywords or None, args.max_occupations)
        count_report[industry] = len(selected)
        one = _build_output(
            selected_occ_df=selected,
            occupations_key=industry,
            source_dir=source_dir,
            top_n=args.top_n,
        )
        occupations_out.update(one["occupations"])

    result = {
        "metadata": {
            "source": "O*NET Online Database",
            "url": "https://www.onetcenter.org/database.html",
            "industries": list(occupations_out.keys()),
            "fields_collected": FIELDS_COLLECTED,
        },
        "occupations": occupations_out,
    }
    total_occupations = sum(len(v) for v in occupations_out.values())

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export complete: {output_path.resolve()} (occupations={total_occupations})")
    for industry, count in count_report.items():
        print(f"  {industry}: {count}")


if __name__ == "__main__":
    main()
