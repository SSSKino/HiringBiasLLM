import argparse
import json
import os
import time
import threading
from pathlib import Path
from typing import Any
from openai import OpenAI


def normalize_industry_label(value: str) -> str:
    """Normalize industry aliases to canonical labels used by the pipeline."""
    raw = (value or "").strip()
    lower = raw.lower()
    if lower in ("law", "legal"):
        return "Legal"
    if lower == "it":
        return "IT"
    if lower == "hr":
        return "HR"
    if lower == "finance":
        return "Finance"
    return raw


def get_occupations_for_industry(occupations_by_industry: dict, target_industry: str) -> list:
    """Get occupations by canonical industry label with alias fallback."""
    if target_industry in occupations_by_industry:
        return occupations_by_industry[target_industry]
    if target_industry == "Legal" and "Law" in occupations_by_industry:
        return occupations_by_industry["Law"]
    return []

class CVScoringUtils:
    """Helper utilities for CV scoring pipeline."""
    
    @staticmethod
    def load_json(path: Path) -> Any:
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON in {path}: {exc}") from exc

    @staticmethod
    def get_JD_INFO(occupations: list[dict], limit: int | None) -> list[dict]:
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

    @staticmethod
    def decompose_JD_to_Batch(items: list[dict], batch_size: int) -> list[list[dict]]:
        if batch_size < 1:
            raise SystemExit("--jobs-batch-size must be >= 1")
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    @staticmethod
    def load_prompt(path: Path) -> dict:
        cfg = CVScoringUtils.load_json(path)
        if not isinstance(cfg, dict):
            raise SystemExit("testing_config.json must be a JSON object")
        required = ["system_msg", "eval_instructions", "output_format", "rubric", "JD_INSTRUCTION"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise SystemExit(f"testing_config.json missing keys: {', '.join(missing)}")
        return cfg

    @staticmethod
    def build_prompt(candidates: list[dict], standard: list[dict], cfg: dict, debug: bool = False) -> list[dict]:
        """构建prompt，将JD、CV、JD_INSTRUCTION、RUBRIC和output_format插入到eval_instructions中。"""
        # 获取JD_INSTRUCTION
        jd_instruction = cfg.get("JD_INSTRUCTION", "")
        
        # 先替换system_msg中的占位符
        system_msg = cfg["system_msg"].replace("[JD_INSTRUCTION]", jd_instruction)

        # 将JD转换为JSON字符串
        jd_content = json.dumps(standard, ensure_ascii=False, indent=2) if standard else ""

        # 将CV处理为JSON列表
        cv_content = json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else ""

        # 将output_format转换为JSON字符串（展示具体的返回格式）
        output_format_content = json.dumps(cfg["output_format"], ensure_ascii=False, indent=2)

        # 将rubric转换为JSON字符串
        rubric_content = json.dumps(cfg.get("rubric", {}), ensure_ascii=False, indent=2)

        # 用实际内容替换placeholder
        eval_instructions = (
            cfg["eval_instructions"]
            .replace("[JD_CONTENT]", jd_content)
            .replace("[CV_CONTENT]", cv_content)
            .replace("[OUTPUT_FORMAT]", output_format_content)
            .replace("[RUBRIC]", rubric_content)
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": eval_instructions},
        ]
        
        # 调试：仅在 debug=True 时保存 prompt 到文件
        if debug:
            debug_file = "debug_prompt.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SYSTEM MESSAGE\n")
                f.write("=" * 80 + "\n")
                f.write(system_msg + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("USER MESSAGE\n")
                f.write("=" * 80 + "\n")
                f.write(eval_instructions + "\n")
            print(f"[DEBUG] Prompt saved to {debug_file}")
        
        return messages

    @staticmethod
    def score(
        client: OpenAI, model: str, candidates: list[dict], standard: list[dict], cfg: dict
    ) -> dict[str, Any]:
        messages = CVScoringUtils.build_prompt(candidates, standard, cfg)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            top_p=1,
        )
        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # fallback: wrap raw content
            return {"raw_response": content}

    @staticmethod
    def check_APIReturn_Format(result: Any) -> bool:
        """验证 result 是否符合预期评分结构（不要求 reasoning）。"""
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
            """检查 scores 字典是否包含所有 6 个维度，且每个维度都有合法 score。"""
            if not isinstance(scores_dict, dict):
                return False

            dict_keys = set(scores_dict.keys())
            if dict_keys != required_dimensions:
                return False

            for dim, dim_data in scores_dict.items():
                if not isinstance(dim_data, dict):
                    return False
                # No-reason prompt: only require score. Original check also required "reasoning".
                if "score" not in dim_data:
                    return False
                # if "score" not in dim_data or "reasoning" not in dim_data:
                #     return False
                # score应该是整数0-100
                score = dim_data.get("score")
                if not isinstance(score, (int, float)) or not (0 <= score <= 100):
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

    @staticmethod
    def mask_CV_KEY(
        cv,
        mask_candidate_id,
        mask_cv_id,
        mask_region=False,
    ):
        masked = dict(cv)

        if mask_candidate_id:
            masked.pop("candidate_id", None)

        if mask_cv_id:
            masked.pop("cv_id", None)

        if mask_region:
            masked.pop("region", None)
            masked.pop("name_category", None)

        return masked

    @staticmethod
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

def run_experiment(
    name: str,
    target_industry: str,
    cv_path: Path,
    occupations_by_industry: dict,
    args: argparse.Namespace,
    client: OpenAI | None,
    review_mode: bool,
    mask_candidate_id: bool,
    mask_cv_id: bool,
    mask_region: bool,
    output_path: Path | None = None,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cvs = CVScoringUtils.load_json(cv_path)
    if not isinstance(cvs, list):
        raise SystemExit(f"CV file must be a JSON list: {cv_path}")

    results: list[dict] = []
    review_payloads: list[dict] = []

    lock = threading.Lock()
    api_call_count = 0
    stream_first_result = True
    stream_file = None

    occupations = get_occupations_for_industry(occupations_by_industry, target_industry)
    if not isinstance(occupations, list):
        raise SystemExit(f"Invalid occupations list for industry: {target_industry}")

    jd_start_index = args.JD_START - 1
    if jd_start_index >= len(occupations):
        raise SystemExit(
            f"--JD_START {args.JD_START} is out of range for industry {target_industry}; "
            f"only {len(occupations)} jobs available"
        )
    selected_occupations = (
        occupations[jd_start_index:]
        if args.JD_NUM is None
        else occupations[jd_start_index : jd_start_index + args.JD_NUM]
    )

    # 结果流式写入：每个结果返回后立即落盘，避免大量结果占用内存
    if (not review_mode) and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_file = output_path.open("w", encoding="utf-8")
        stream_file.write('{\n')
        stream_file.write(f'  "industry": {json.dumps(target_industry, ensure_ascii=False)},\n')
        stream_file.write('  "experiments": [\n')
        stream_file.write('    {\n')
        stream_file.write(f'      "name": {json.dumps(name, ensure_ascii=False)},\n')
        stream_file.write(f'      "cv_path": {json.dumps(str(cv_path), ensure_ascii=False)},\n')
        stream_file.write(f'      "mask_candidate_id": {json.dumps(mask_candidate_id)},\n')
        stream_file.write(f'      "mask_cv_id": {json.dumps(mask_cv_id)},\n')
        stream_file.write(f'      "mask_region": {json.dumps(mask_region)},\n')
        stream_file.write('      "results": [\n')
        stream_file.flush()

    print(f"[{name}] Using multi-threaded processing (5s staggered start)")

    # ------------------------
    # Worker（线程执行函数）
    # ------------------------
    def score_single_jd_cv_pair(cv: dict, cv_idx: int, occupation: dict, jd_index: int):
        nonlocal api_call_count, stream_first_result

        industry = normalize_industry_label(str(cv.get("industry_target", "NA")))
        if industry != target_industry:
            return

        candidate_id = cv.get("candidate_id")
        cv_id = cv.get("cv_id", candidate_id)

        masked_cv = CVScoringUtils.mask_CV_KEY(cv, mask_candidate_id=mask_candidate_id, mask_cv_id=mask_cv_id, mask_region=mask_region)
        single_job_standard = CVScoringUtils.get_JD_INFO([occupation], limit=1)
        jd_info = {
            "jd_index": jd_index,
            "jd_soc_code": occupation.get("soc_code"),
            "jd_title": occupation.get("title"),
        }

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
                        **jd_info,
                    }
                )
                # 为第一个请求构建并保存 prompt（用于调试）
                if len(review_payloads) == 1:
                    CVScoringUtils.build_prompt([masked_cv], single_job_standard, args.prompt_config, debug=True)
            return

        print(f"[{name}] ⚙️ {timestamp} API #{current_call} → CV[{cv_idx}] scoring...")

        # ✅ 复用 client（关键）
        result = CVScoringUtils.score(client, args.model, [masked_cv], single_job_standard, args.prompt_config)

        print(f"[{name}] ⚙️ {timestamp} API #{current_call} ← returned")

        # No-reason prompt: skip retry block because outputs intentionally omit "reasoning".
        # retry with debug info
        # if not CVScoringUtils.check_APIReturn_Format(result):
        #     # 打印调试信息
        #     print(f"[{name}] ⚙️ {timestamp} API #{current_call} → format check failed, result: {json.dumps(result, ensure_ascii=False)[:200]}...")
        #     print(f"[{name}] ⚙️ {timestamp} API #{current_call} → retrying...")
        #     result = CVScoringUtils.score(client, args.model, [masked_cv], single_job_standard, args.prompt_config)
        #     print(f"[{name}] ⚙️ {timestamp} API #{current_call} ← retry returned")
        #
        #     # 检查 retry 后的格式
        #     if not CVScoringUtils.check_APIReturn_Format(result):
        #         print(f"[{name}] ⚙️ {timestamp} API #{current_call} ⚠️ WARNING: retry also failed format check")

        result = CVScoringUtils.attach_ids(result, candidate_id, cv_id)
        result["industry"] = industry
        result["region"] = cv.get("region")
        result.update(jd_info)

        with lock:
            if stream_file is not None:
                if not stream_first_result:
                    stream_file.write(',\n')
                stream_file.write("        " + json.dumps(result, ensure_ascii=False))
                stream_file.flush()
                stream_first_result = False
            else:
                results.append(result)

    # ------------------------
    # 构建任务列表
    # ------------------------
    tasks = []
    processed_cv_count = 0

    for cv_idx, cv in enumerate(cvs):
        if not isinstance(cv, dict):
            continue

        cv_industry = normalize_industry_label(str(cv.get("industry_target", "NA")))
        if cv_industry != target_industry:
            continue

        if args.CV_NUM is not None and processed_cv_count >= args.CV_NUM:
            break

        processed_cv_count += 1

        for local_jd_offset, occupation in enumerate(selected_occupations):
            jd_index = jd_start_index + local_jd_offset + 1
            tasks.append((cv, cv_idx, occupation, jd_index))

    print(f"[{name}] Total tasks: {len(tasks)}")

    # ------------------------
    # 多线程执行（带5秒间隔）
    # ------------------------
    start_time = time.time()

    max_workers = args.thread

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for task in tasks:
                future = executor.submit(score_single_jd_cv_pair, *task)
                futures.append(future)

                # ⭐ 每个任务启动间隔5秒
                time.sleep(5)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"[{name}] Worker failed: {exc}")

    except KeyboardInterrupt:
        print(f"\n[{name}] ⚠️ Interrupted by user")

    elapsed = time.time() - start_time

    if stream_file is not None:
        stream_file.write('\n      ]\n')
        stream_file.write('    }\n')
        stream_file.write('  ]\n')
        stream_file.write('}\n')
        stream_file.close()

    print(f"[{name}] ────────────────────────────────────────")
    completed_count = len(results) if stream_file is None else api_call_count
    print(f"[{name}] Completed: {completed_count} API calls")
    print(f"[{name}] Total time: {elapsed:.2f}s")

    return {
        "name": name,
        "cv_path": str(cv_path),
        "mask_candidate_id": mask_candidate_id,
        "mask_cv_id": mask_cv_id,
        "mask_region": mask_region,
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
        "--prompt",
        default=r".\testing_config.json",
        help="Path to prompt json file.",
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
        "--JD_START",
        type=int,
        default=1,
        help="1-based start position in the selected industry's JD list (default: 1).",
    )
    parser.add_argument(
        "--industry",
        default="IT",
        help="Industry to score, or 'all' for every industry (default: IT).",
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
    parser.add_argument(
        "--thread",
        type=int,
        default=1,
        help="Number of worker threads for parallel API calls (default: 1).",
    )
    args = parser.parse_args()

    if args.JD_START < 1:
        raise SystemExit("--JD_START must be >= 1")

    args.prompt_config = CVScoringUtils.load_prompt(Path(args.prompt))

    jobs_data = CVScoringUtils.load_json(Path(args.jobs))
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

    def sanitize_filename_part(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)

    # Pre-calc counts for confirmation
    if args.industry.lower() == "all":
        selected_industries = [
            normalize_industry_label(name)
            for name, occs in occupations_by_industry.items()
            if isinstance(occs, list) and occs
        ]
        # 去重并保持顺序
        selected_industries = list(dict.fromkeys(selected_industries))
        if not selected_industries:
            raise SystemExit("No valid industries found in jobs dataset")
    else:
        target_industry = normalize_industry_label(args.industry)
        selected_industries = [target_industry]

    for target_industry in selected_industries:
        occupations = get_occupations_for_industry(occupations_by_industry, target_industry)
        if not isinstance(occupations, list) or not occupations:
            raise SystemExit(f"Invalid occupations list for industry: {target_industry}")
        if args.JD_START > len(occupations):
            raise SystemExit(
                f"--JD_START {args.JD_START} is out of range for industry {target_industry}; "
                f"only {len(occupations)} jobs available"
            )

    experiments = []
    if args.experiment in ("1", "all"):
        experiments.append(
            {
                "key": "exp1",
                "name": "exp1_mask_candidate_and_cv_nonimplicit",
                "cv_path": Path(args.cv),
                "mask_candidate_id": True,
                "mask_cv_id": True,
                "mask_region": True,
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
                "mask_region": True,
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
                "mask_region": False,  # EXP3: region is included as an explicit feature.
            }
        )

    if len(selected_industries) == 1:
        print(f"Industry: {selected_industries[0]}")
    else:
        print(f"Industry: all ({len(selected_industries)} industries)")
    jd_end = "all remaining" if args.JD_NUM is None else args.JD_START + args.JD_NUM - 1
    print(f"JD selection: start={args.JD_START}, end={jd_end}")
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
            for target_industry in selected_industries:
                industry_tag = sanitize_filename_part(target_industry)
                for exp in experiments:
                    default_out_path = Path(f"{args.output_prefix}_{industry_tag}_{exp['key']}{run_suffix}.json")
                    can_use_base_output = (
                        base_output
                        and args.experiment != "all"
                        and args.runs == 1
                        and len(selected_industries) == 1
                    )
                    out_path = base_output if can_use_base_output else default_out_path

                    exp_result = run_experiment(
                        name=exp["name"],
                        target_industry=target_industry,
                        cv_path=exp["cv_path"],
                        occupations_by_industry=occupations_by_industry,
                        args=args,
                        client=run_client,
                        review_mode=review_mode,
                        mask_candidate_id=exp["mask_candidate_id"],
                        mask_cv_id=exp["mask_cv_id"],
                        mask_region=exp["mask_region"],
                        output_path=None if review_mode else out_path,
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
                        default_review_path = Path(
                            f"{args.output_prefix}_{industry_tag}_{exp['key']}{run_suffix}_review.json"
                        )
                        review_out = base_output if can_use_base_output else default_review_path
                        Path(review_out).write_text(
                            json.dumps(exp_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        print(
                            f"[{target_industry}] Wrote review payloads to {Path(review_out).resolve()}"
                        )
                    else:
                        print(
                            f"[{target_industry}] Wrote experiment results to {Path(out_path).resolve()}"
                        )
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user (Ctrl+C)")
        print("Saved partial results if any.")
        return

    return

if __name__ == "__main__":
    main()

# python e:\code\py\hiringbias\main_testing.py --experiment 1 --industry IT --JD_NUM 1 --CV_NUM 3 --model qwen3.5-flash --api-key sk-4uMGXOuP1nWUbHBF1Wz4MQ --config e:\code\py\hiringbias\testing_config.json
