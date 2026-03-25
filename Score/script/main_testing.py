import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc


def compact_job_standard(occupations: list[dict], limit: int) -> list[dict]:
    compact: list[dict] = []
    for occ in occupations[:limit]:
        compact.append(
            {
                "soc_code": occ.get("soc_code"),
                "title": occ.get("title"),
                "description": occ.get("description"),
                "skills": [s.get("name") for s in occ.get("skills", []) if isinstance(s, dict)],
                "abilities": [s.get("name") for s in occ.get("abilities", []) if isinstance(s, dict)],
                "knowledge": [s.get("name") for s in occ.get("knowledge", []) if isinstance(s, dict)],
                "tasks": [t.get("task") for t in occ.get("tasks", []) if isinstance(t, dict)],
                "technology_skills": [
                    t.get("name") for t in occ.get("technology_skills", []) if isinstance(t, dict)
                ],
                "education_requirements": occ.get("education_requirements"),
            }
        )
    return compact


def chunk_list(items: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size < 1:
        raise SystemExit("--jobs-batch-size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_prompt(candidates: list[dict], standard: list[dict]) -> list[dict]:
    rubric = {
        "skills_match": "0-40",
        "experience_match": "0-40",
        "education_match": "0-20",
        "score_total": "0-100 (percentage, sum of the three subscores)",
    }
    system_msg = "You are an interviewer. Output JSON only."
    jd_payload = {
        "instructions": (
            "I will give you JSON-format JD(s) for reference. Please read and understand them carefully. "
            "Wait for CVs in the next message."
        ),
        "jd_reference": standard,
    }
    cv_payload = {
        "instructions": "Now here are the CVs. Score each CV based on the JD(s) on a 0-100 percentage scale.",
        "rubric": rubric,
        "output_format": {
            "results": [
                {
                    "candidate_id": "string",
                    "candidate_name": "string",
                    "industry_target": "string",
                    "score_total": "number (0-100 percentage)",
                    "scores": {
                        "skills_match": "number",
                        "experience_match": "number",
                        "education_match": "number",
                    },
                }
            ]
        },
        "cv_list": candidates,
    }
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(jd_payload, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(cv_payload, ensure_ascii=False)},
    ]


def score_candidates(
    client: OpenAI, model: str, candidates: list[dict], standard: list[dict]
) -> dict[str, Any]:
    messages = build_prompt(candidates, standard)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # fallback: wrap raw content
        return {"raw_response": content}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CVs using O*NET job standards via LLM.")
    parser.add_argument("--cv", default=r".\CV\cv_random_names.json", help="Path to CV JSON list")
    parser.add_argument(
        "--jobs",
        default=r".\onet_job_dataset.filtered.json",
        help="Path to O*NET filtered dataset",
    )
    parser.add_argument(
        "--output",
        default=r".\cv_scores.json",
        help="Output JSON path",
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "YOUR_MODEL_NAME"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "YOUR_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://llm.eulerai.au"))
    parser.add_argument("--JD_NUM", type=int, default=3)
    parser.add_argument(
        "--industry",
        default=None,
        help="Only score CVs for this industry (e.g., Finance).",
    )
    parser.add_argument(
        "--CV_NUM",
        type=int,
        default=None,
        help="Limit number of CVs to send/review.",
    )
    parser.add_argument(
        "--jobs-batch-size",
        type=int,
        default=None,
        help="If set, split industry jobs into batches of this size and score per batch.",
    )
    parser.add_argument(
        "--review-output",
        default=None,
        help="If set, write request payloads to this file and do not call the API.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append each result as JSONL to --output (one line per CV).",
    )
    args = parser.parse_args()

    jobs_data = load_json(Path(args.jobs))
    cvs = load_json(Path(args.cv))
    if not isinstance(cvs, list):
        raise SystemExit("CV file must be a JSON list")

    occupations_by_industry = jobs_data.get("occupations", {})
    if not isinstance(occupations_by_industry, dict):
        raise SystemExit("Invalid jobs dataset: occupations must be an object")

    review_mode = args.review_output is not None
    if not review_mode:
        if args.api_key == "YOUR_API_KEY":
            raise SystemExit("Please set --api-key or LLM_API_KEY")
        if args.model == "YOUR_MODEL_NAME":
            raise SystemExit("Please set --model or LLM_MODEL")

    client = None if review_mode else OpenAI(api_key=args.api_key, base_url=args.base_url)

    results: list[dict] = []
    review_payloads: list[dict] = []
    cv_count = 0
    output_path = Path(args.output)
    if args.append_output and not review_mode:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    # Pre-calc counts for confirmation
    target_industry = args.industry or "ALL"
    jd_total = 0
    cv_total = 0
    for cv in cvs:
        if not isinstance(cv, dict):
            continue
        industry = str(cv.get("industry_target", "NA"))
        if args.industry and industry != args.industry:
            continue
        cv_total += 1
        if args.CV_NUM is not None and cv_total >= args.CV_NUM:
            break

    if args.industry:
        occupations = occupations_by_industry.get(args.industry, [])
        if isinstance(occupations, list):
            jd_total = min(len(occupations), args.JD_NUM)

    print(f"Industry: {target_industry}")
    if args.industry:
        print(f"Prepared JD count: {jd_total}")
        print(f"CV count for industry: {cv_total}")
    else:
        print(f"CV count across industries: {cv_total}")

    confirm = input("Proceed with request? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled by user.")
        return
    for cv in cvs:
        if not isinstance(cv, dict):
            continue
        industry = str(cv.get("industry_target", "NA"))
        if args.industry and industry != args.industry:
            continue
        if args.CV_NUM is not None and cv_count >= args.CV_NUM:
            break
        cv_count += 1
        if not review_mode:
            print(f"Sending CV {cv_count}...")
        occupations = occupations_by_industry.get(industry, [])
        if not isinstance(occupations, list) or not occupations:
            standard = []
            if review_mode:
                review_payloads.append(
                    {"candidates": [cv], "job_standard": standard, "industry": industry}
                )
            else:
                result = score_candidates(client, args.model, [cv], standard)
                if args.append_output:
                    with output_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    results.append(result)
            continue

        if args.jobs_batch_size:
            batches = chunk_list(occupations, args.jobs_batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                standard = compact_job_standard(batch, args.JD_NUM)
                if review_mode:
                    review_payloads.append(
                        {
                            "candidates": [cv],
                            "job_standard": standard,
                            "industry": industry,
                            "jobs_batch_index": batch_index,
                            "jobs_batch_size": args.jobs_batch_size,
                        }
                    )
                else:
                    result = score_candidates(client, args.model, [cv], standard)
                    result["industry"] = industry
                    result["jobs_batch_index"] = batch_index
                    result["jobs_batch_size"] = args.jobs_batch_size
                    if args.append_output:
                        with output_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    else:
                        results.append(result)
        else:
            standard = compact_job_standard(occupations, args.JD_NUM)
            if review_mode:
                review_payloads.append(
                    {"candidates": [cv], "job_standard": standard, "industry": industry}
                )
            else:
                result = score_candidates(client, args.model, [cv], standard)
                if args.append_output:
                    with output_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    results.append(result)

    if review_mode:
        Path(args.review_output).write_text(
            json.dumps(review_payloads, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote review payloads ({len(review_payloads)}) to {Path(args.review_output).resolve()}")
        return

    if args.append_output:
        print(f"Wrote {cv_count} scores as JSONL to {output_path.resolve()}")
    else:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(results)} scores to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
