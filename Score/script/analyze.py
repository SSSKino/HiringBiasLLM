import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any
import statistics
from glob import glob
import matplotlib.pyplot as plt
import textwrap

# Ensure readable axis labels on Windows
plt.rcParams["font.family"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_candidate_id(candidate_id: str) -> tuple[str, str, str]:
    """Parse candidate_id into (industry, seniority, region)."""
    parts = candidate_id.split("_")
    if len(parts) >= 3:
        region = "_".join(parts[2:]).replace("_", " ")
        return parts[0], parts[1], region
    if len(parts) == 2:
        return parts[0], parts[1], "NA"
    return "NA", "NA", "NA"


def normalize_region(value: str) -> str:
    """Normalize region names."""
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


def load_scores(path: Path, experiment: str | None, score_glob: str) -> dict[str, dict]:
    """Load scores from JSON files (handles nested structure)."""
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
        
        # Treat each JSON file as a separate experiment based on filename
        file_exp_key = p.stem
        
        if isinstance(data, dict) and "experiments" in data:
            # If file has nested experiments structure, use filename as experiment key
            # and combine all experiments from this file
            results_by_exp.setdefault(file_exp_key, {"scores": [], "sources": set()})
            results_by_exp[file_exp_key]["sources"].add(p.stem)
            
            for exp in data.get("experiments", []):
                if not isinstance(exp, dict):
                    continue
                
                for item in exp.get("results", []):
                    if isinstance(item, dict):
                        results_by_exp[file_exp_key]["scores"].append(item)
        elif isinstance(data, list):
            # List format - treat entire file as one experiment
            results_by_exp.setdefault(file_exp_key, {"scores": [], "sources": set()})
            results_by_exp[file_exp_key]["sources"].add(p.stem)
            for item in data:
                if isinstance(item, dict):
                    results_by_exp[file_exp_key]["scores"].append(item)
        elif isinstance(data, dict) and "scores" in data:
            # Dictionary with scores field - treat as one experiment
            results_by_exp.setdefault(file_exp_key, {"scores": [], "sources": set()})
            results_by_exp[file_exp_key]["sources"].add(p.stem)
            scores_field = data.get("scores", [])
            if isinstance(scores_field, list):
                for item in scores_field:
                    if isinstance(item, dict):
                        results_by_exp[file_exp_key]["scores"].append(item)

    return results_by_exp


def normalize_record(r: dict) -> dict:
    """Normalize a result record.
    
    Each dimension score is 0-100. Total score is average of 6 dimensions.
    Save all scores normalized to 0-1 range.
    """
    dimensions = [
        "skill_match",
        "experience_match",
        "education_match",
        "communication_and_collaboration",
        "execution_compliance_reliability",
        "role_context_adaptability",
    ]

    scores = {}

    # Extract scores from the scores field
    scores_dict = r.get("scores", {})
    if isinstance(scores_dict, dict):
        for dim in dimensions:
            dim_data = scores_dict.get(dim)

            # Case 1: Standard format {"score": xx, "reasoning": "..."}
            if isinstance(dim_data, dict):
                score = dim_data.get("score")
            # Case 2: Direct number (backward compatibility)
            elif isinstance(dim_data, (int, float)):
                score = dim_data
            else:
                continue

            # Only keep valid scores (0-100)
            if isinstance(score, (int, float)):
                scores[dim] = float(score)

    # Calculate total score (average of dimensions)
    if scores:
        total_score = sum(scores.values()) / 6.0
    else:
        total_score = 0.0

    # Create normalized result
    normalized = {
        "candidate_id": r.get("candidate_id", "NA"),
        "cv_id": r.get("cv_id", "NA"),
        "industry": r.get("industry", "NA"),
    }
    
    # Add normalized dimension scores (divide by 6)
    for dim in dimensions:
        if dim in scores:
            normalized[dim] = round(scores[dim] / 6.0, 4)
    
    # Add total normalized score (divide by 6)
    normalized["score_total_normalized"] = round(total_score, 4)

    return normalized


def analyze_stability(normalized_results: list[dict]) -> dict:
    """Analyze scores grouped by region."""
    
    # Group normalized scores by region
    region_scores: dict[str, list[float]] = defaultdict(list)
    region_candidates: dict[str, set] = defaultdict(set)
    result_count = len(normalized_results)
    
    print(f"Processing {result_count} normalized results...")
    
    for result in normalized_results:
        candidate_id = str(result.get("candidate_id", "NA"))
        _, _, region = parse_candidate_id(candidate_id)
        region = normalize_region(region)
        
        normalized_score = result.get("score_total_normalized")
        
        if isinstance(normalized_score, (int, float)):
            region_scores[region].append(float(normalized_score))
            region_candidates[region].add(candidate_id)
    
    print(f"Found {len(region_scores)} regions\n")
    
    # Calculate statistics by region
    stability_report = {
        "total_results": result_count,
        "unique_regions": len(region_scores),
        "regions": {}
    }
    
    for region in sorted(region_scores.keys()):
        scores = region_scores[region]
        candidates_count = len(region_candidates[region])
        
        if scores:
            stability_report["regions"][region] = {
                "scores": [round(s, 2) for s in scores],
                "count": len(scores),
                "unique_candidates": candidates_count,
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
                "mean": round(statistics.mean(scores), 2),
                "median": round(statistics.median(scores), 2),
                "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0
            }
    
    return stability_report


def print_summary(report: dict) -> None:
    """Print a summary grouped by region."""
    print("\n" + "="*90)
    print("SCORE ANALYSIS BY REGION (Normalized Data)")
    print("="*90)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total results: {report['total_results']}")
    print(f"  Unique regions: {report['unique_regions']}")
    
    print(f"\n📈 Scores by Region:")
    print(f"{'Region':<40} {'Count':<8} {'Candidates':<12} {'Mean':<10} {'Median':<10} {'Std Dev':<10}")
    print("-" * 90)
    
    for region in sorted(report["regions"].keys()):
        stats = report["regions"][region]
        region_display = region[:39] if len(region) <= 39 else region[:36] + "..."
        print(f"{region_display:<40} {stats['count']:<8} {stats['unique_candidates']:<12} {stats['mean']:<10.2f} {stats['median']:<10.2f} {stats['stdev']:<10.2f}")
    
    print("\n" + "="*90 + "\n")


def generate_stability_charts(report: dict, scores_by_exp: dict, normalized_dir: Path, output_dir: Path, avg_mode: bool = False) -> int:
    """Generate visualization charts for normalized scores analysis.
    
    Args:
        report: Analysis report dictionary
        scores_by_exp: Dictionary mapping experiment names to their normalized results
        normalized_dir: Directory containing normalized JSON files
        output_dir: Directory to save generated charts
        avg_mode: If False (default), generate per-file region charts; if True, generate single averaged chart
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = Path(normalized_dir)
    chart_count = 0
    
    regions_fixed = [
        "Sub-Saharan Africa",
        "Northern Africa and Western Asia",
        "Central and Southern Asia",
        "Eastern and South-Eastern Asia",
        "Latin America and the Caribbean",
        "Oceania",
        "Europe and Northern America",
    ]
    
    color_map = {
        "Sub-Saharan Africa": '#1f77b4',
        "Northern Africa and Western Asia": '#ff7f0e',
        "Central and Southern Asia": '#2ca02c',
        "Eastern and South-Eastern Asia": '#d62728',
        "Latin America and the Caribbean": '#9467bd',
        "Oceania": '#8c564b',
        "Europe and Northern America": '#e377c2',
    }
    
    # Collect all normalized results for dimension chart (always use all data)
    all_normalized_results = []
    for exp_name, exp_normalized_list in scores_by_exp.items():
        # scores_by_exp contains lists of normalized results
        if isinstance(exp_normalized_list, list):
            all_normalized_results.extend(exp_normalized_list)
    
    if "regions" not in report:
        return chart_count
    
    # Chart 1: Average Scores by Region
    if avg_mode:
        # Single averaged chart across all data
        fig, ax = plt.subplots(figsize=(14, 7))
        
        region_labels = []
        region_means = []
        colors_list = []
        
        for region in regions_fixed:
            if region in report["regions"]:
                mean_score = report["regions"][region]["mean"]
                region_labels.append(region)
                region_means.append(mean_score)
                colors_list.append(color_map.get(region, '#1f77b4'))
        
        if region_labels:
            bars = ax.bar(range(len(region_labels)), region_means, color=colors_list, edgecolor='black', linewidth=1.2, alpha=0.8)
            
            for i, (bar, mean_val) in enumerate(zip(bars, region_means)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(region_labels)))
            ax.set_xticklabels([textwrap.fill(label, width=12) for label in region_labels], fontsize=10)
            
            ax.set_ylabel("Average Normalized Score", fontsize=12, fontweight='bold')
            ax.set_xlabel("Region", fontsize=12, fontweight='bold')
            ax.set_title("Average Scores by Region (All Data)", fontsize=13, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        chart_path = output_dir / "scores_by_region_avg.png"
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {chart_path.name}")
        chart_count += 1
    else:
        # Per-file region charts - generate one for each experiment
        print(f"Generating {len(scores_by_exp)} per-file region charts...")
        for exp_name, exp_normalized_list in scores_by_exp.items():
            # exp_normalized_list is directly the list of normalized results
            if not isinstance(exp_normalized_list, list) or not exp_normalized_list:
                print(f"  ⚠️ Skipping {exp_name}: no data")
                continue
            
            # Analyze region scores for this specific experiment
            region_scores = defaultdict(list)
            
            for result in exp_normalized_list:
                candidate_id = str(result.get("candidate_id", "NA"))
                _, _, region = parse_candidate_id(candidate_id)
                region = normalize_region(region)
                
                normalized_score = result.get("score_total_normalized")
                if isinstance(normalized_score, (int, float)):
                    region_scores[region].append(float(normalized_score))
            
            # Calculate means for this experiment
            fig, ax = plt.subplots(figsize=(14, 7))
            
            region_labels = []
            region_means = []
            colors_list = []
            
            for region in regions_fixed:
                if region in region_scores and region_scores[region]:
                    mean_score = statistics.mean(region_scores[region])
                    region_labels.append(region)
                    region_means.append(mean_score)
                    colors_list.append(color_map.get(region, '#1f77b4'))
            
            if region_labels:
                bars = ax.bar(range(len(region_labels)), region_means, color=colors_list, edgecolor='black', linewidth=1.2, alpha=0.8)
                
                for i, (bar, mean_val) in enumerate(zip(bars, region_means)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_xticks(range(len(region_labels)))
                ax.set_xticklabels([textwrap.fill(label, width=12) for label in region_labels], fontsize=10)
                
                ax.set_ylabel("Average Normalized Score", fontsize=12, fontweight='bold')
                ax.set_xlabel("Region", fontsize=12, fontweight='bold')
                ax.set_title(f"Average Scores by Region ({exp_name})", fontsize=13, fontweight='bold')
                ax.set_ylim(0, 105)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            chart_path = output_dir / f"scores_by_region_{exp_name}.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Generated: {chart_path.name}")
            chart_count += 1
    
    # Chart 2: Score Difference (Max - Min) by Dimension, grouped by region
    # Data format: For each dimension/total_score across ALL normalized JSON files,
    # we calculate: for each region, max_score - min_score in that region
    # Then display the maximum difference found across all regions, with region label
    # Example: If we have 3 normalized JSONs with region scores:
    #   json1: region1=50, region2=30, region3=40
    #   json2: region1=40, region2=90, region3=30
    #   json3: region1=10, region2=80, region3=50
    # Then: region1_diff=40, region2_diff=60, region3_diff=20
    # Chart shows max_diff=60 from region2
    fig, ax = plt.subplots(figsize=(16, 10))
    
    dimensions = [
        "skill_match",
        "experience_match",
        "education_match",
        "communication_and_collaboration",
        "execution_compliance_reliability",
        "role_context_adaptability",
    ]
    
    dimension_labels = [
        "Skill Match",
        "Experience Match",
        "Education Match",
        "Communication & Collaboration",
        "Execution & Reliability",
        "Role Context & Adaptability",
    ]
    
    # Group results by region
    region_grouped_results = defaultdict(list)
    for result in all_normalized_results:
        candidate_id = str(result.get("candidate_id", "NA"))
        _, _, region = parse_candidate_id(candidate_id)
        region = normalize_region(region)
        region_grouped_results[region].append(result)
    
    # For each dimension and total score, calculate max-min within each region, then get the maximum across regions
    dimension_metadata = []  # List of dicts with diff info, region, source files, and values
    display_labels = []
    
    # Calculate dimension differences
    for i, dim in enumerate(dimensions):
        max_diff_across_regions = 0
        max_diff_source_region = "NA"
        max_score_value = 0
        min_score_value = 0
        max_score_source_file = "NA"
        min_score_source_file = "NA"
        
        for region, region_results in region_grouped_results.items():
            dim_scores_with_source = []  # List of (score, source_file, result)
            
            for result in region_results:
                # Directly extract dimension score (no longer nested in 'scores' dict)
                dim_score = result.get(dim)
                
                if isinstance(dim_score, (int, float)):
                    source_file = result.get("_source_file", "Unknown")
                    dim_scores_with_source.append((float(dim_score), source_file, result))
            
            # Calculate max-min for this dimension in this region
            if dim_scores_with_source:
                max_entry = max(dim_scores_with_source, key=lambda x: x[0])
                min_entry = min(dim_scores_with_source, key=lambda x: x[0])
                region_diff = max_entry[0] - min_entry[0]
                
                if region_diff > max_diff_across_regions:
                    max_diff_across_regions = region_diff
                    max_diff_source_region = region
                    max_score_value = max_entry[0]
                    min_score_value = min_entry[0]
                    max_score_source_file = max_entry[1]
                    min_score_source_file = min_entry[1]
        
        if max_diff_across_regions >= 0:
            dimension_metadata.append({
                'diff': max_diff_across_regions,
                'region': max_diff_source_region,
                'max_file': max_score_source_file,
                'min_file': min_score_source_file,
                'max_value': max_score_value,
                'min_value': min_score_value
            })
            display_labels.append(dimension_labels[i])
    
    # Calculate total score difference (per region, then max across regions)
    max_total_diff_across_regions = 0
    max_total_diff_source_region = "NA"
    max_total_score_value = 0
    min_total_score_value = 0
    max_total_score_source_file = "NA"
    min_total_score_source_file = "NA"
    
    for region, region_results in region_grouped_results.items():
        total_scores_with_source = []  # List of (total_score, source_file, result)
        
        for result in region_results:
            # Use score_total_normalized directly, but also divide by 6 to match dimension scale
            total_score = result.get("score_total_normalized")
            if isinstance(total_score, (int, float)):
                source_file = result.get("_source_file", "Unknown")
                # Divide by 6 to match the dimension scale (0-16.67)
                normalized_total = float(total_score) / 6.0
                total_scores_with_source.append((normalized_total, source_file, result))
        
        # Calculate max-min for total score in this region
        if total_scores_with_source:
            max_entry = max(total_scores_with_source, key=lambda x: x[0])
            min_entry = min(total_scores_with_source, key=lambda x: x[0])
            region_diff = max_entry[0] - min_entry[0]
            
            if region_diff > max_total_diff_across_regions:
                max_total_diff_across_regions = region_diff
                max_total_diff_source_region = region
                max_total_score_value = max_entry[0]
                min_total_score_value = min_entry[0]
                max_total_score_source_file = max_entry[1]
                min_total_score_source_file = min_entry[1]
    
    if max_total_diff_across_regions >= 0:
        dimension_metadata.append({
            'diff': max_total_diff_across_regions,
            'region': max_total_diff_source_region,
            'max_file': max_total_score_source_file,
            'min_file': min_total_score_source_file,
            'max_value': max_total_score_value,
            'min_value': min_total_score_value
        })
        display_labels.append("Total Score")
    
    # Create bar chart
    x = range(len(display_labels))
    colors = ['steelblue'] * len(dimensions) + ['red']
    
    # Extract diff values for plotting
    dimension_diffs = [meta['diff'] for meta in dimension_metadata]
    
    bars = ax.bar(x, dimension_diffs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels with region and source file information for each bar
    for i, (bar, meta) in enumerate(zip(bars, dimension_metadata)):
        height = bar.get_height()
        
        # Add difference value label at the top
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.15,
               f'{meta["diff"]:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add source file information for max and min - compact table format
        max_file = meta['max_file'].replace('_normalized', '').replace('cv_scores_', '')
        min_file = meta['min_file'].replace('_normalized', '').replace('cv_scores_', '')
        max_val = meta['max_value']
        min_val = meta['min_value']
        
        # Compact format in a box
        source_info = f"Max: {max_val:.2f} ({max_file})\nMin: {min_val:.2f} ({min_file})"
        
        # Add box closer to bar with better positioning
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1.5,
               source_info, ha='center', va='bottom', fontsize=9, color='white',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='steelblue', alpha=0.85, edgecolor='darkblue', linewidth=1.2))
        
        # Add region label closer to source info (reduce gap)
        region_display = textwrap.fill(meta['region'], width=12)
        ax.text(bar.get_x() + bar.get_width() / 2., height + 2.6,
               region_display, ha='center', va='bottom', fontsize=9, style='italic', color='darkred', fontweight='bold')
    
    # Set x-axis labels with text wrapping
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(label, width=12) for label in display_labels], fontsize=10)
    
    ax.set_ylabel("Score Difference (Max - Min)", fontsize=12, fontweight='bold')
    ax.set_title("Score Variability by Dimension", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(dimension_diffs) * 1.45 if dimension_diffs else 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / "score_variability_by_segment.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {chart_path.name}")
    chart_count += 1
    
    return chart_count


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize normalized scores by region from multiple score files."
    )
    parser.add_argument(
        "--input",
        default=r".",
        help="Path to scores JSON or a directory of score JSON files",
    )
    parser.add_argument(
        "--score-glob",
        default="*score*.json",
        help="Glob pattern for score files when input is a directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file for detailed report (JSON format).",
    )
    parser.add_argument(
        "--chart-dir",
        default=r".\stability_charts",
        help="Output directory for visualization charts.",
    )
    parser.add_argument(
        "--normalized-dir",
        default=r".\normalized",
        help="Output directory for normalized JSON files.",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="If set, generate single averaged region chart across all data; otherwise generate per-file region charts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    normalized_dir = Path(args.normalized_dir)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all score files
    scores_by_exp = load_scores(input_path, None, args.score_glob)
    
    if not scores_by_exp:
        raise SystemExit("No scores found in input.")
    
    print(f"Loaded {len(scores_by_exp)} experiments\n")
    
    # Collect all results and normalize them, save normalized files
    all_normalized_results = []
    scores_by_exp_normalized = {}  # Store normalized results for each experiment
    
    for exp_name, exp_data in scores_by_exp.items():
        raw_scores = exp_data.get("scores", [])
        print(f"Normalizing {len(raw_scores)} results from '{exp_name}'...")
        
        exp_normalized_results = []
        for result in raw_scores:
            normalized = normalize_record(result)
            # Add source file information for tracking in charts (internal use only)
            normalized["_source_file"] = exp_name
            all_normalized_results.append(normalized)
            exp_normalized_results.append(normalized)
        
        # Store normalized results for this experiment
        scores_by_exp_normalized[exp_name] = exp_normalized_results
        
        # Save normalized results for this experiment (WITHOUT _source_file field)
        norm_file = normalized_dir / f"{exp_name}_normalized.json"
        results_to_save = [
            {k: v for k, v in result.items() if k != "_source_file"}
            for result in exp_normalized_results
        ]
        norm_file.write_text(
            json.dumps(results_to_save, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  → Saved normalized results to {norm_file}")
    
    print(f"\nTotal normalized results: {len(all_normalized_results)}\n")
    
    # Analyze by region
    report = analyze_stability(all_normalized_results)
    
    # Print summary
    print_summary(report)
    
    # Generate charts
    chart_dir = Path(args.chart_dir)
    chart_count = generate_stability_charts(report, scores_by_exp_normalized, normalized_dir, chart_dir, avg_mode=args.avg)
    print(f"\n📊 Generated {chart_count} chart(s) to {chart_dir.resolve()}\n")
    
    # Save detailed report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Detailed report written to {output_path}")
    
    print(f"✅ Normalized JSON files saved to {normalized_dir.resolve()}")


if __name__ == "__main__":
    main()
