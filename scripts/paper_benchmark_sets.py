#!/usr/bin/env python3
"""Current paper benchmark groups and legacy compatibility aliases."""

from __future__ import annotations

from typing import Any

from extended_20_optimization_problems import EXTENDED_20_OPTIMIZATION_PROBLEMS
from extended_optimization_problems import EXTENDED_OPTIMIZATION_PROBLEMS
from paper_general_hard_problems import PAPER_GENERAL_HARD_PROBLEMS
from paper_general_numeric_problems import PAPER_GENERAL_NUMERIC_PROBLEMS
from paper_pareval_problems import PAPER_PAREVAL_PROBLEMS
from paper_probe_problems import PAPER_PROBE_PROBLEMS
from paper_source_numerical_problems import PAPER_SOURCE_NUMERICAL_PROBLEMS
from paper_source_system_problems import PAPER_SOURCE_SYSTEM_PROBLEMS
from paper_tailcase_problems import PAPER_TAILCASE_PROBLEMS
from simple_optimization_problems import SIMPLE_OPTIMIZATION_PROBLEMS


def select_named_problems(problems: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    wanted = set(names)
    selected = [problem for problem in problems if problem["name"] in wanted]
    if len(selected) != len(names):
        found = {problem["name"] for problem in selected}
        missing = [name for name in names if name not in found]
        raise ValueError(f"Missing paper benchmark problems: {missing}")
    order = {name: idx for idx, name in enumerate(names)}
    return sorted(selected, key=lambda problem: order[problem["name"]])


PAPER_EASY4_PROBLEMS = (
    select_named_problems(EXTENDED_OPTIMIZATION_PROBLEMS, ["matrix_transpose_cache"])
    + select_named_problems(SIMPLE_OPTIMIZATION_PROBLEMS, ["array_sum_unroll"])
    + select_named_problems(EXTENDED_20_OPTIMIZATION_PROBLEMS, ["heap_sort_implementation"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["csr_spmv_axpy_dot"])
)

PAPER_MEDIUM4_PROBLEMS = (
    select_named_problems(EXTENDED_OPTIMIZATION_PROBLEMS, ["vector_add_simd"])
    + select_named_problems(SIMPLE_OPTIMIZATION_PROBLEMS, ["max_value_branchless"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["rmsnorm_mixed"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["floyd_warshall_blocked"])
)

PAPER_HARD4_PROBLEMS = (
    select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["radix_sort_u32_pairs", "conv2d_3x3_multi_channel"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["stencil3d_mixed_7pt"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["nbody_tiled_step"])
)

PAPER12_PROBLEMS = PAPER_EASY4_PROBLEMS + PAPER_MEDIUM4_PROBLEMS + PAPER_HARD4_PROBLEMS

PAPER_COMPETITIVE4_PROBLEMS = (
    select_named_problems(SIMPLE_OPTIMIZATION_PROBLEMS, ["array_sum_unroll"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["rmsnorm_mixed"])
    + select_named_problems(SIMPLE_OPTIMIZATION_PROBLEMS, ["max_value_branchless"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["stencil3d_mixed_7pt"])
)

PAPER_TAILCASE4_PROBLEMS = select_named_problems(
    PAPER_TAILCASE_PROBLEMS,
    [
        "stencil2d_halo5",
        "transpose_strided_f64",
        "csr_spmv_alpha_beta",
        "segmented_prefix_sum_i32",
    ],
)

PAPER_STRIDE4_PROBLEMS = select_named_problems(
    PAPER_TAILCASE_PROBLEMS,
    [
        "transpose_strided_f64",
        "crop2d_strided_u8",
        "gather_rows_strided_f64",
        "scatter_cols_strided_f32",
    ],
)

PAPER_NUMERIC2_PROBLEMS = select_named_problems(
    PAPER_TAILCASE_PROBLEMS,
    [
        "gemm_strided_alpha_beta_f64",
        "lower_tri_solve_strided_f64",
    ],
)

PAPER_GENERAL4_PROBLEMS = select_named_problems(
    PAPER_GENERAL_HARD_PROBLEMS,
    [
        "banded_edit_distance_i32",
        "softmax_cross_entropy_stable_f32",
        "csr_pagerank_step_f64",
        "utf8_validate_strict",
    ],
)

PAPER_GENERAL_NUMERIC1_PROBLEMS = select_named_problems(
    PAPER_GENERAL_HARD_PROBLEMS,
    [
        "cholesky_spd_f64",
    ],
)

PAPER_GENERAL_SELECTED2_PROBLEMS = select_named_problems(
    PAPER_GENERAL_HARD_PROBLEMS,
    [
        "utf8_validate_strict",
        "cholesky_spd_f64",
    ],
)

PAPER_GENERAL_NUMERIC4_PROBLEMS = select_named_problems(
    PAPER_GENERAL_NUMERIC_PROBLEMS,
    [
        "tridiagonal_solve_f64",
        "natural_cubic_spline_f64",
        "quadratic_roots_stable_f64",
        "eig2x2_symmetric_f64",
    ],
)

PAPER_PAREVAL4_PROBLEMS = select_named_problems(
    PAPER_PAREVAL_PROBLEMS,
    [
        "pareval_convex_hull_perimeter_f64",
        "pareval_largest_component_i32",
        "pareval_sort_ignore_zero_i32",
        "pareval_game_of_life_i32",
    ],
)

PAPER_PROBE_MIXED1_PROBLEMS = select_named_problems(
    PAPER_PROBE_PROBLEMS,
    [
        "mixed_precision_dot_i8_f32",
    ],
)

PAPER_PROBE_NEW4_PROBLEMS = select_named_problems(
    PAPER_PROBE_PROBLEMS,
    [
        "percent_decode_strict",
        "csv_parse_quoted_fields",
        "roi_crop_clamp_u8",
        "topk_ignore_nan_f32",
    ],
)

PAPER_SMALLMODEL_PILOT5_PROBLEMS = (
    select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["utf8_validate_strict"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_sort_ignore_zero_i32"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["crop2d_strided_u8"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["lower_tri_solve_strided_f64"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["radix_sort_u32_pairs"])
)

PAPER_SMALLMODEL_K10_PILOT3_PROBLEMS = (
    select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_sort_ignore_zero_i32"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["crop2d_strided_u8"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["radix_sort_u32_pairs"])
)

PAPER_9B_K10_SELECTED20_PROBLEMS = (
    select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["banded_edit_distance_i32"])
    + select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["cholesky_spd_f64"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["conv2d_3x3_multi_channel"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["crop2d_strided_u8"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["csr_spmv_axpy_dot"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["floyd_warshall_blocked"])
    + select_named_problems(EXTENDED_20_OPTIMIZATION_PROBLEMS, ["heap_sort_implementation"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["lower_tri_solve_strided_f64"])
    + select_named_problems(EXTENDED_OPTIMIZATION_PROBLEMS, ["matrix_transpose_cache"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_convex_hull_perimeter_f64"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_largest_component_i32"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_sort_ignore_zero_i32"])
    + select_named_problems(PAPER_GENERAL_NUMERIC_PROBLEMS, ["quadratic_roots_stable_f64"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["radix_sort_u32_pairs"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["rmsnorm_mixed"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["stencil2d_halo5"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["stencil3d_mixed_7pt"])
    + select_named_problems(PAPER_PROBE_PROBLEMS, ["topk_ignore_nan_f32"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["transpose_strided_f64"])
    + select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["utf8_validate_strict"])
)

PAPER_SENSITIVITY2_PROBLEMS = (
    select_named_problems(PAPER_TAILCASE_PROBLEMS, ["transpose_strided_f64"])
    + select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["utf8_validate_strict"])
)

PAPER_SENSITIVITY4_PROBLEMS = (
    select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_sort_ignore_zero_i32"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["stencil3d_mixed_7pt"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["crop2d_strided_u8"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["floyd_warshall_blocked"])
)

PAPER_SELECTED12_PROBLEMS = (
    select_named_problems(
        PAPER_TAILCASE_PROBLEMS,
        [
            "crop2d_strided_u8",
            "transpose_strided_f64",
        ],
    )
    + select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["cholesky_spd_f64"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_convex_hull_perimeter_f64"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["radix_sort_u32_pairs"])
    + select_named_problems(PAPER_SOURCE_SYSTEM_PROBLEMS, ["floyd_warshall_blocked"])
    + select_named_problems(EXTENDED_20_OPTIMIZATION_PROBLEMS, ["heap_sort_implementation"])
    + select_named_problems(PAPER_PAREVAL_PROBLEMS, ["pareval_sort_ignore_zero_i32"])
    + select_named_problems(PAPER_TAILCASE_PROBLEMS, ["lower_tri_solve_strided_f64"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["stencil3d_mixed_7pt"])
    + select_named_problems(PAPER_SOURCE_NUMERICAL_PROBLEMS, ["rmsnorm_mixed"])
    + select_named_problems(PAPER_GENERAL_HARD_PROBLEMS, ["utf8_validate_strict"])
)


BENCHMARKS = {
    "simple5": SIMPLE_OPTIMIZATION_PROBLEMS,
    "extended10": EXTENDED_OPTIMIZATION_PROBLEMS,
    "paper_easy4": PAPER_EASY4_PROBLEMS,
    "paper_medium4": PAPER_MEDIUM4_PROBLEMS,
    "paper_hard4": PAPER_HARD4_PROBLEMS,
    "paper_competitive4": PAPER_COMPETITIVE4_PROBLEMS,
    "paper_tailcase4": PAPER_TAILCASE4_PROBLEMS,
    "paper_stride4": PAPER_STRIDE4_PROBLEMS,
    "paper_numeric2": PAPER_NUMERIC2_PROBLEMS,
    "paper_general4": PAPER_GENERAL4_PROBLEMS,
    "paper_general_numeric1": PAPER_GENERAL_NUMERIC1_PROBLEMS,
    "paper_general_numeric4": PAPER_GENERAL_NUMERIC4_PROBLEMS,
    "paper_general_selected2": PAPER_GENERAL_SELECTED2_PROBLEMS,
    "paper_pareval4": PAPER_PAREVAL4_PROBLEMS,
    "paper_probe_mixed1": PAPER_PROBE_MIXED1_PROBLEMS,
    "paper_probe_new4": PAPER_PROBE_NEW4_PROBLEMS,
    "paper_9b_k10_selected20": PAPER_9B_K10_SELECTED20_PROBLEMS,
    "paper_smallmodel_k10_pilot3": PAPER_SMALLMODEL_K10_PILOT3_PROBLEMS,
    "paper_smallmodel_pilot5": PAPER_SMALLMODEL_PILOT5_PROBLEMS,
    "paper_sensitivity2": PAPER_SENSITIVITY2_PROBLEMS,
    "paper_sensitivity4": PAPER_SENSITIVITY4_PROBLEMS,
    "paper_selected12": PAPER_SELECTED12_PROBLEMS,
    "paper12": PAPER12_PROBLEMS,
}


LEGACY_BENCHMARK_ALIASES = {
    "paper_easy": "paper_easy4",
    "paper_medium": "paper_medium4",
    "paper_hard": "paper_hard4",
}
