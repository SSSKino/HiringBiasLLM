import argparse
import json
import os
import time
import threading
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


def compact_job_standard(occupations: list[dict], limit: int | None) -> list[dict]:
    """Compact job occupations. If limit is None, use all occupations."""
    compact: list[dict] = []
    # 如果 limit 是 None，表示使用全部；否则用 slice
    target_occupations = occupations if limit is None else occupations[:limit]
    for occ in target_occupations:
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
    required = ["system_msg", "jd_instructions", "cv_instructions", "rubric"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"testing_config.json missing keys: {', '.join(missing)}")
    return cfg


def build_prompt(candidates: list[dict], standard: list[dict], cfg: dict) -> list[dict]:
    """构建prompt，将JD和CV内容插入到instructions中。"""
    system_msg = cfg["system_msg"]
    
    # 将JD转换为JSON字符串
    jd_content = json.dumps(standard, ensure_ascii=False, indent=2) if standard else ""
    
    # 将CV处理为JSON列表
    cv_content = json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else ""
    
    # 用实际内容替换placeholder
    jd_instructions = cfg["jd_instructions"].replace("[JD_CONTENT]", jd_content)
    cv_instructions = cfg["cv_instructions"].replace("[CV_CONTENT]", cv_content)
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": jd_instructions},
        {"role": "user", "content": cv_instructions},
    ]


def score_candidates(
    client: OpenAI, model: str, candidates: list[dict], standard: list[dict], cfg: dict
) -> dict[str, Any]:
    messages = build_prompt(candidates, standard, cfg)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        top_p=1,
    )
    content = resp.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # fallback: wrap raw content
        return {"raw_response": content}


def has_expected_score_shape(result: Any) -> bool:
    """验证result是否有预期的评分结构（按照 testing_config.json 中的格式）。"""
    # 预期的6维度（小写+下划线）
    required_dimensions = {
        "skill_match",
        "experience_match", 
        "education_match",
        "communication_and_collaboration",
        "execution_compliance_reliability",
        "role_context_adaptability"
    }
    
    def check_score_structure(scores_dict: dict) -> bool:
        """检查scores字典是否包含所有6个维度，每个维度都有score、reasoning、evidence。"""
        if not isinstance(scores_dict, dict):
            return False
        
        dict_keys = set(scores_dict.keys())
        if dict_keys != required_dimensions:
            return False
        
        for dim, dim_data in scores_dict.items():
            if not isinstance(dim_data, dict):
                return False
            # 检查必需字段：score、reasoning、evidence
            if "score" not in dim_data or "reasoning" not in dim_data or "evidence" not in dim_data:
                return False
            # score应该是整数0-100
            score = dim_data.get("score")
            if not isinstance(score, (int, float)) or not (0 <= score <= 100):
                return False
            # evidence应该是列表
            if not isinstance(dim_data.get("evidence"), list):
                return False
        return True
    
    # 格式1：直接的 scores 对象（顶层是 scores）
    if check_score_structure(result):
        return True
    
    # 格式2：包含 scores 字段的对象（如 {"candidate_id": "...", "scores": {...}, ...}）
    if isinstance(result, dict) and "scores" in result:
        if check_score_structure(result["scores"]):
            return True
    
    # 格式3：嵌套的结果列表
    if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
        if not result["results"]:
            return False
        for item in result["results"]:
            if isinstance(item, dict) and "scores" in item:
                if not check_score_structure(item["scores"]):
                    return False
        return True
    
    # 格式4：直接的结果列表
    if isinstance(result, list):
        if not result:
            return False
        for item in result:
            if isinstance(item, dict) and "scores" in item:
                if not check_score_structure(item["scores"]):
                    return False
        return True
    
    return False


def sanitize_cv(cv: dict, mask_candidate_id: bool, mask_cv_id: bool) -> dict:
    masked = dict(cv)
    if mask_candidate_id and "candidate_id" in masked:
        masked.pop("candidate_id")
    if mask_cv_id and "cv_id" in masked:
        masked.pop("cv_id")
    return masked


def attach_ids(result: Any, candidate_id: str | None, cv_id: str | None) -> Any:
    if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
        for item in result["results"]:
            if isinstance(item, dict):
                if cv_id is not None:
                    item["cv_id"] = cv_id
                if candidate_id is not None:
                    item["candidate_id"] = candidate_id
        return result
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                if cv_id is not None:
                    item["cv_id"] = cv_id
                if candidate_id is not None:
                    item["candidate_id"] = candidate_id
        return result
    if isinstance(result, dict):
        if cv_id is not None:
            result["cv_id"] = cv_id
        if candidate_id is not None:
            result["candidate_id"] = candidate_id
        return result
    return result

# def run_experiment(
#     name: str,
#     cv_path: Path,
#     occupations_by_industry: dict,
#     args: argparse.Namespace,
#     client: OpenAI | None,
#     review_mode: bool,
#     mask_candidate_id: bool,
#     mask_cv_id: bool,
# ) -> dict:
#     cvs = load_json(cv_path)
#     if not isinstance(cvs, list):
#         raise SystemExit(f"CV file must be a JSON list: {cv_path}")

#     results: list[dict] = []
#     review_payloads: list[dict] = []
#     api_call_count = 0

#     target_industry = args.industry
#     occupations = occupations_by_industry.get(target_industry, [])
#     if not isinstance(occupations, list):
#         raise SystemExit(f"Invalid occupations list for industry: {target_industry}")

#     def score_single_jd_cv_pair(cv: dict, cv_idx: int, occupation: dict, jd_index: int, total_jds: int) -> None:
#         """为单个 JD+CV 对进行评分。单线程处理。"""
#         nonlocal api_call_count
        
#         industry = str(cv.get("industry_target", "NA"))
        
#         if industry != target_industry:
#             return
        
#         candidate_id = cv.get("candidate_id")
#         cv_id = cv.get("cv_id", candidate_id)
#         masked_cv = sanitize_cv(cv, mask_candidate_id=mask_candidate_id, mask_cv_id=mask_cv_id)
        
#         # 每个occupation单独compact成一个包含1个元素的列表
#         single_job_standard = compact_job_standard([occupation], limit=1)
        
#         api_call_count += 1
#         timestamp = time.strftime('%H:%M:%S')
        
#         if review_mode:
#             review_payloads.append(
#                 {
#                     "candidates": [masked_cv],
#                     "job_standard": single_job_standard,
#                     "industry": industry,
#                     "jd_index": jd_index,
#                     "jd_total": total_jds,
#                 }
#             )
#         else:
#             print(f"[{name}] ⚙️ {timestamp} API #{api_call_count} → CV[{cv_idx}] + JD[{jd_index}]...")
#             # 为每对1JD+1CV创建新客户端连接
#             temp_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
#             result = score_candidates(temp_client, args.model, [masked_cv], single_job_standard, args.prompt_config)
#             print(f"[{name}] ⚙️ {timestamp} API #{api_call_count} ← returned")
#             if not has_expected_score_shape(result):
#                 print(f"[{name}] ⚙️ {timestamp} API #{api_call_count} → retrying...")
#                 # 重试时创建新客户端
#                 temp_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
#                 result = score_candidates(temp_client, args.model, [masked_cv], single_job_standard, args.prompt_config)
#                 print(f"[{name}] ⚙️ {timestamp} API #{api_call_count} ← retry returned")
#             result = attach_ids(result, candidate_id, cv_id)
#             result["industry"] = industry
#             result["jd_index"] = jd_index
#             result["jd_total"] = total_jds
#             results.append(result)

#     # 相关JD数量
#     selected_occupations = occupations if args.JD_NUM is None else occupations[:args.JD_NUM]
#     total_jds = len(selected_occupations)
    
#     # 预先计算会实际使用的CV数量（只计算行业匹配的）
#     matching_cv_count = 0
#     for cv in cvs:
#         if isinstance(cv, dict):
#             cv_industry = str(cv.get("industry_target", "NA"))
#             if cv_industry == target_industry:
#                 matching_cv_count += 1
#                 if args.CV_NUM is not None and matching_cv_count >= args.CV_NUM:
#                     break
    
#     # 单线程处理 (CV, JD) 对
#     print(f"[{name}] Total CVs loaded: {len(cvs)}")
#     print(f"[{name}] Target industry: {target_industry}")
#     print(f"[{name}] CVs matching industry: {matching_cv_count}")
#     print(f"[{name}] JD count: {total_jds}")
#     print(f"[{name}] Actual CV+JD pairs to process: {matching_cv_count} × {total_jds} = {matching_cv_count * total_jds}")
#     print(f"[{name}] Using single-threaded sequential processing")
#     print(f"[{name}] ────────────────────────────────────────")
    
#     start_time = time.time()
#     try:
#         processed_cv_count = 0
#         for cv_idx, cv in enumerate(cvs):
#             if not isinstance(cv, dict):
#                 continue
#             # 检查行业匹配
#             cv_industry = str(cv.get("industry_target", "NA"))
#             if cv_industry != target_industry:
#                 continue
#             # 检查CV数量限制
#             if args.CV_NUM is not None and processed_cv_count >= args.CV_NUM:
#                 break
#             processed_cv_count += 1
            
#             # 为这个CV以及所有JD依次处理
#             for jd_index, occupation in enumerate(selected_occupations, start=1):
#                 score_single_jd_cv_pair(cv, cv_idx, occupation, jd_index, total_jds)
#     except KeyboardInterrupt:
#         print(f"\n[{name}] ⚠️ Interrupted by user (Ctrl+C)")
#         print(f"[{name}] Partial results: {len(results)} API calls")
    
#     elapsed = time.time() - start_time
#     print(f"[{name}] ────────────────────────────────────────")
#     print(f"[{name}] Completed: {processed_cv_count} CVs × {total_jds} JDs = {len(results)} API calls")
#     print(f"[{name}] Total time: {elapsed:.2f}s")
#     if len(results) > 0:
#         print(f"[{name}] Average per API call: {elapsed/len(results):.2f}s")

#     return {
#         "name": name,
#         "cv_path": str(cv_path),
#         "mask_candidate_id": mask_candidate_id,
#         "mask_cv_id": mask_cv_id,
#         "results": results,
#         "review_payloads": review_payloads,
#     }
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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cvs = load_json(cv_path)
    if not isinstance(cvs, list):
        raise SystemExit(f"CV file must be a JSON list: {cv_path}")

    results: list[dict] = []
    review_payloads: list[dict] = []

    lock = threading.Lock()
    api_call_count = 0

    target_industry = args.industry
    occupations = occupations_by_industry.get(target_industry, [])
    if not isinstance(occupations, list):
        raise SystemExit(f"Invalid occupations list for industry: {target_industry}")

    selected_occupations = occupations if args.JD_NUM is None else occupations[:args.JD_NUM]
    total_jds = len(selected_occupations)

    print(f"[{name}] Using multi-threaded processing (5s staggered start)")

    # ------------------------
    # Worker（线程执行函数）
    # ------------------------
    def score_single_jd_cv_pair(cv: dict, cv_idx: int, occupation: dict, jd_index: int):
        nonlocal api_call_count

        industry = str(cv.get("industry_target", "NA"))
        if industry != target_industry:
            return

        candidate_id = cv.get("candidate_id")
        cv_id = cv.get("cv_id", candidate_id)

        masked_cv = sanitize_cv(cv, mask_candidate_id=mask_candidate_id, mask_cv_id=mask_cv_id)
        single_job_standard = compact_job_standard([occupation], limit=1)

        with lock:
            api_call_count += 1
            current_call = api_call_count

        timestamp = time.strftime('%H:%M:%S')

        if review_mode:
            with lock:
                review_payloads.append(
                    {
                        "candidates": [masked_cv],
                        "job_standard": single_job_standard,
                        "industry": industry,
                        "jd_index": jd_index,
                        "jd_total": total_jds,
                    }
                )
            return

        print(f"[{name}] ⚙️ {timestamp} API #{current_call} → CV[{cv_idx}] + JD[{jd_index}]...")

        # ✅ 复用 client（关键）
        result = score_candidates(client, args.model, [masked_cv], single_job_standard, args.prompt_config)

        print(f"[{name}] ⚙️ {timestamp} API #{current_call} ← returned")

        # retry
        if not has_expected_score_shape(result):
            print(f"[{name}] ⚙️ {timestamp} API #{current_call} → retrying...")
            result = score_candidates(client, args.model, [masked_cv], single_job_standard, args.prompt_config)
            print(f"[{name}] ⚙️ {timestamp} API #{current_call} ← retry returned")

        result = attach_ids(result, candidate_id, cv_id)
        result["industry"] = industry
        result["jd_index"] = jd_index
        result["jd_total"] = total_jds

        with lock:
            results.append(result)

    # ------------------------
    # 构建任务列表
    # ------------------------
    tasks = []
    processed_cv_count = 0

    for cv_idx, cv in enumerate(cvs):
        if not isinstance(cv, dict):
            continue

        cv_industry = str(cv.get("industry_target", "NA"))
        if cv_industry != target_industry:
            continue

        if args.CV_NUM is not None and processed_cv_count >= args.CV_NUM:
            break

        processed_cv_count += 1

        for jd_index, occupation in enumerate(selected_occupations, start=1):
            tasks.append((cv, cv_idx, occupation, jd_index))

    print(f"[{name}] Total tasks: {len(tasks)}")

    # ------------------------
    # 多线程执行（带5秒间隔）
    # ------------------------
    start_time = time.time()

    max_workers = 5

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for task in tasks:
                future = executor.submit(score_single_jd_cv_pair, *task)
                futures.append(future)

                # ⭐ 每个任务启动间隔5秒
                time.sleep(5)

            for future in as_completed(futures):
                pass

    except KeyboardInterrupt:
        print(f"\n[{name}] ⚠️ Interrupted by user")

    elapsed = time.time() - start_time

    print(f"[{name}] ────────────────────────────────────────")
    print(f"[{name}] Completed: {len(results)} API calls")
    print(f"[{name}] Total time: {elapsed:.2f}s")

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
    # 支持 --JD_NUM N 或 --JD_NUM all
    def parse_jd_num(value: str):
        if value.lower() == 'all':
            return None  # None 表示全部
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--JD_NUM must be an integer or 'all', got '{value}'")
    
    parser.add_argument("--JD_NUM", type=parse_jd_num, default=3,
                        help="Number of top jobs to use for scoring, or 'all' for all jobs (default: 3)")
    parser.add_argument(
        "--industry",
        default="IT",
        help="Only score CVs for this industry (default: IT).",
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
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs. Each run uses a new client and writes a separate output file.",
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

    def new_client() -> OpenAI:
        return OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Pre-calc counts for confirmation
    target_industry = args.industry
    jd_total = 0
    occupations = occupations_by_industry.get(args.industry, [])
    if isinstance(occupations, list):
        if args.JD_NUM is None:
            jd_total = len(occupations)  # 使用全部
        else:
            jd_total = min(len(occupations), args.JD_NUM)
    else:
        raise SystemExit(f"Invalid occupations list for industry: {args.industry}")

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

    try:
        for run_idx in range(1, args.runs + 1):
            run_client = None if review_mode else new_client()
            run_suffix = f"_run{run_idx}"
            for exp in experiments:
                exp_result = run_experiment(
                    name=exp["name"],
                    cv_path=exp["cv_path"],
                    occupations_by_industry=occupations_by_industry,
                    args=args,
                    client=run_client,
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
                        if base_output and args.experiment != "all" and args.runs == 1
                        else Path(f"{args.output_prefix}_{exp['key']}{run_suffix}_review.json")
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
                        if base_output and args.experiment != "all" and args.runs == 1
                        else Path(f"{args.output_prefix}_{exp['key']}{run_suffix}.json")
                    )
                    Path(out_path).write_text(
                        json.dumps(exp_output, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"Wrote experiment results to {Path(out_path).resolve()}")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user (Ctrl+C)")
        print("Saved partial results if any.")
        return

    return


if __name__ == "__main__":
    main()
