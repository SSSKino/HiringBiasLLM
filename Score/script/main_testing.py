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


def load_testing_config(path: Path) -> dict:
    cfg = load_json(path)
    if not isinstance(cfg, dict):
        raise SystemExit("testing_config.json must be a JSON object")
    required = ["system_msg", "jd_instructions", "cv_instructions", "rubric", "output_format"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"testing_config.json missing keys: {', '.join(missing)}")
    return cfg


def build_prompt(candidates: list[dict], standard: list[dict], cfg: dict) -> list[dict]:
    system_msg = cfg["system_msg"]
    jd_payload = {
        "instructions": cfg["jd_instructions"],
        "jd_reference": standard,
    }
    cv_payload = {
        "instructions": cfg["cv_instructions"],
        "rubric": cfg["rubric"],
        "output_format": cfg["output_format"],
        "cv_list": candidates,
    }
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(jd_payload, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(cv_payload, ensure_ascii=False)},
    ]


def score_candidates(
    client: OpenAI, model: str, candidates: list[dict], standard: list[dict], cfg: dict
) -> dict[str, Any]:
    messages = build_prompt(candidates, standard, cfg)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # fallback: wrap raw content
        return {"raw_response": content}


def sanitize_cv(cv: dict, mask_candidate_id: bool, mask_cv_id: bool) -> dict:
    masked = dict(cv)
    if mask_candidate_id and "candidate_id" in masked:
        masked.pop("candidate_id")
    if mask_cv_id and "cv_id" in masked:
        masked.pop("cv_id")
    return masked


def attach_ids(result: dict, candidate_id: str | None, cv_id: str | None) -> dict:
    if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
        for item in result["results"]:
            if isinstance(item, dict):
                if cv_id is not None:
                    item["cv_id"] = cv_id
                if candidate_id is not None:
                    item["candidate_id"] = candidate_id
        return result
    # fallback: wrap ids at top-level
    if cv_id is not None:
        result["cv_id"] = cv_id
    if candidate_id is not None:
        result["candidate_id"] = candidate_id
    return result


def run_experiment(
    name: str,
    cv_path: Path,
    occupations_by_industry: dict,
    args: argparse.Namespace,
    client: OpenAI | None,
    review_mode: bool,
    mask_candidate_id: bool,
    mask_cv_id: bool,
) -> dict:
    cvs = load_json(cv_path)
    if not isinstance(cvs, list):
        raise SystemExit(f"CV file must be a JSON list: {cv_path}")

    results: list[dict] = []
    review_payloads: list[dict] = []
    cv_count = 0

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
            print(f"[{name}] Sending CV {cv_count}...")

        occupations = occupations_by_industry.get(industry, [])
        candidate_id = cv.get("candidate_id")
        cv_id = cv.get("cv_id", candidate_id)
        masked_cv = sanitize_cv(cv, mask_candidate_id=mask_candidate_id, mask_cv_id=mask_cv_id)

        if not isinstance(occupations, list) or not occupations:
            standard = []
            if review_mode:
                review_payloads.append(
                    {"candidates": [masked_cv], "job_standard": standard, "industry": industry}
                )
            else:
                result = score_candidates(client, args.model, [masked_cv], standard, args.prompt_config)
                result = attach_ids(result, candidate_id, cv_id)
                results.append(result)
            continue

        if args.jobs_batch_size:
            batches = chunk_list(occupations, args.jobs_batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                standard = compact_job_standard(batch, args.JD_NUM)
                if review_mode:
                    review_payloads.append(
                        {
                            "candidates": [masked_cv],
                            "job_standard": standard,
                            "industry": industry,
                            "jobs_batch_index": batch_index,
                            "jobs_batch_size": args.jobs_batch_size,
                        }
                    )
                else:
                    result = score_candidates(client, args.model, [masked_cv], standard, args.prompt_config)
                    result = attach_ids(result, candidate_id, cv_id)
                    result["industry"] = industry
                    result["jobs_batch_index"] = batch_index
                    result["jobs_batch_size"] = args.jobs_batch_size
                    results.append(result)
        else:
            standard = compact_job_standard(occupations, args.JD_NUM)
            if review_mode:
                review_payloads.append(
                    {"candidates": [masked_cv], "job_standard": standard, "industry": industry}
                )
            else:
                result = score_candidates(client, args.model, [masked_cv], standard, args.prompt_config)
                result = attach_ids(result, candidate_id, cv_id)
                results.append(result)

    return {
        "name": name,
        "cv_path": str(cv_path),
        "mask_candidate_id": mask_candidate_id,
        "mask_cv_id": mask_cv_id,
        "results": results,
        "review_payloads": review_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CVs using O*NET job standards via LLM.")
    parser.add_argument("--cv", default=r".\CV\cv_random_names.json", help="Path to CV JSON list")
    parser.add_argument(
        "--cv-implicit",
        default=r".\CV\cv_random_name_implicit.json",
        help="Path to implicit CV JSON list",
    )
    parser.add_argument(
        "--jobs",
        default=r".\onet_job_dataset.filtered.json",
        help="Path to O*NET filtered dataset",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (used for single experiment).",
    )
    parser.add_argument(
        "--output-prefix",
        default="cv_scores",
        help="Prefix for per-experiment outputs when running multiple experiments.",
    )
    parser.add_argument(
        "--config",
        default=r".\testing_config.json",
        help="Path to testing_config.json for prompt and rubric.",
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
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["1", "2", "3", "all"],
        help="Which experiment to run: 1, 2, 3, or all (runs sequentially).",
    )
    args = parser.parse_args()

    args.prompt_config = load_testing_config(Path(args.config))

    jobs_data = load_json(Path(args.jobs))
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

    # Pre-calc counts for confirmation
    target_industry = args.industry or "ALL"
    jd_total = 0
    if args.industry:
        occupations = occupations_by_industry.get(args.industry, [])
        if isinstance(occupations, list):
            jd_total = min(len(occupations), args.JD_NUM)

    experiments = []
    if args.experiment in ("1", "all"):
        experiments.append(
            {
                "key": "exp1",
                "name": "exp1_mask_candidate_and_cv_nonimplicit",
                "cv_path": Path(args.cv),
                "mask_candidate_id": True,
                "mask_cv_id": True,
            }
        )
    if args.experiment in ("2", "all"):
        experiments.append(
            {
                "key": "exp2",
                "name": "exp2_mask_candidate_and_cv_implicit",
                "cv_path": Path(args.cv_implicit),
                "mask_candidate_id": True,
                "mask_cv_id": True,
            }
        )
    if args.experiment in ("3", "all"):
        experiments.append(
            {
                "key": "exp3",
                "name": "exp3_mask_cv_only_implicit",
                "cv_path": Path(args.cv_implicit),
                "mask_candidate_id": False,
                "mask_cv_id": True,
            }
        )

    print(f"Industry: {target_industry}")
    if args.industry:
        print(f"Prepared JD count: {jd_total}")
    print(f"Experiments: {[e['name'] for e in experiments]}")

    confirm = input("Proceed with request? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled by user.")
        return

    base_output = Path(args.output) if args.output else None
    all_results = []
    all_review_payloads = []

    for exp in experiments:
        exp_result = run_experiment(
            name=exp["name"],
            cv_path=exp["cv_path"],
            occupations_by_industry=occupations_by_industry,
            args=args,
            client=client,
            review_mode=review_mode,
            mask_candidate_id=exp["mask_candidate_id"],
            mask_cv_id=exp["mask_cv_id"],
        )
        if review_mode:
            exp_payload = {
                "industry": target_industry,
                "experiments": [
                    {
                        "name": exp_result["name"],
                        "cv_path": exp_result["cv_path"],
                        "payloads": exp_result["review_payloads"],
                    }
                ],
            }
            review_out = (
                base_output
                if base_output and args.experiment != "all"
                else Path(f"{args.output_prefix}_{exp['key']}_review.json")
            )
            Path(review_out).write_text(
                json.dumps(exp_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Wrote review payloads to {Path(review_out).resolve()}")
        else:
            exp_output = {
                "industry": target_industry,
                "experiments": [
                    {
                        "name": exp_result["name"],
                        "cv_path": exp_result["cv_path"],
                        "mask_candidate_id": exp_result["mask_candidate_id"],
                        "mask_cv_id": exp_result["mask_cv_id"],
                        "results": exp_result["results"],
                    }
                ],
            }
            out_path = (
                base_output
                if base_output and args.experiment != "all"
                else Path(f"{args.output_prefix}_{exp['key']}.json")
            )
            Path(out_path).write_text(
                json.dumps(exp_output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Wrote experiment results to {Path(out_path).resolve()}")

    return


if __name__ == "__main__":
    main()


