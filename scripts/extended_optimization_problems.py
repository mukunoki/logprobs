"""
Extended Performance Optimization Problems
より多様な性能最適化タスク（10問）

カテゴリ:
- ループ最適化（既存5問）
- アルゴリズム改善（2問）
- 並列化・SIMD（3問）
"""

# 既存5問をインポート
from simple_optimization_problems import SIMPLE_OPTIMIZATION_PROBLEMS

# 新規5問を追加
NEW_OPTIMIZATION_PROBLEMS = [
    # アルゴリズム改善カテゴリ
    {
        "name": "binary_search_opt",
        "description": "二分探索の活用（線形探索→二分探索）",
        "category": "algorithm",
        "optimization_prompt": """Implement an optimized array search in C using binary search.
Function signature: int binary_search(int* arr, int n, int target)

The input array is SORTED in ascending order. Use binary search algorithm:
- Start with left=0, right=n-1
- Compare target with middle element
- Adjust search range based on comparison
- Return index if found, -1 if not found

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int binary_search(int* arr, int n, int target);

int main() {
    int n = 1000000;
    int* arr = malloc(n * sizeof(int));

    // Create sorted array
    for (int i = 0; i < n; i++) {
        arr[i] = i * 2;
    }

    int target = 999998;  // Known to exist at index 499999

    // Warmup
    int result = binary_search(arr, n, target);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 1000; iter++) {
        result = binary_search(arr, n, target);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 1000.0;

    free(arr);

    if (result == 499999) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%d expected=499999\\n", result);
        return 1;
    }
}
""",
        "performance_threshold": 1.0  # ms
    },
    {
        "name": "selection_sort_to_quicksort",
        "description": "ソートアルゴリズム改善（選択ソート→クイックソート）",
        "category": "algorithm",
        "optimization_prompt": """Implement an optimized integer sorting in C using quicksort.
Function signature: void quicksort(int* arr, int n)

Use quicksort algorithm with:
- Partition using last element as pivot
- Recursive sorting of sub-arrays
- In-place sorting (no extra arrays)

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void quicksort(int* arr, int n);

int main() {
    int n = 50000;
    int* arr = malloc(n * sizeof(int));
    int* original = malloc(n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
        original[i] = arr[i];
    }

    // Warmup
    quicksort(arr, n);

    // Restore
    for (int i = 0; i < n; i++) arr[i] = original[i];

    // Measure
    clock_t start = clock();
    quicksort(arr, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Verify
    int sorted = 1;
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i+1]) {
            sorted = 0;
            break;
        }
    }

    free(arr);
    free(original);

    if (sorted) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 50.0  # ms
    },

    # 並列化・SIMDカテゴリ
    {
        "name": "vector_add_simd",
        "description": "ベクトル加算のSIMD最適化",
        "category": "simd",
        "optimization_prompt": """Implement an optimized vector addition in C with SIMD hints.
Function signature: void vector_add(double* a, double* b, double* c, int n)

Compute c[i] = a[i] + b[i] for all i. Use optimization techniques:
- Loop unrolling by 4 or 8
- Multiple accumulators for instruction-level parallelism
- Compiler auto-vectorization hints

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void vector_add(double* a, double* b, double* c, int n);

int main() {
    int n = 10000000;
    double* a = malloc(n * sizeof(double));
    double* b = malloc(n * sizeof(double));
    double* c = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    // Warmup
    vector_add(a, b, c, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        vector_add(a, b, c, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    // Verify
    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(c[i] - 3.0) > 0.001) {
            correct = 0;
            break;
        }
    }

    free(a);
    free(b);
    free(c);

    if (correct) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 30.0  # ms
    },
    {
        "name": "matrix_transpose_cache",
        "description": "行列転置のキャッシュ最適化",
        "category": "simd",
        "optimization_prompt": """Implement an optimized matrix transpose in C with cache blocking.
Function signature: void matrix_transpose(int n, double A[n][n], double B[n][n])

Transpose matrix A into B (B[j][i] = A[i][j]). Use cache optimization:
- Block/tile the matrix (e.g., 32x32 or 64x64 blocks)
- Process each block to improve cache locality
- Minimize cache misses

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_transpose(int n, double A[n][n], double B[n][n]);

int main() {
    int n = 512;
    double (*A)[n] = malloc(sizeof(double[n][n]));
    double (*B)[n] = malloc(sizeof(double[n][n]));

    // Initialize
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i * n + j;
            B[i][j] = 0.0;
        }
    }

    // Warmup
    matrix_transpose(n, A, B);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 5; iter++) {
        matrix_transpose(n, A, B);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 5.0;

    // Verify
    int correct = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (B[j][i] != A[i][j]) {
                correct = 0;
                goto done;
            }
        }
    }
done:
    free(A);
    free(B);

    if (correct) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 100.0  # ms
    },
    {
        "name": "prefix_sum_parallel",
        "description": "累積和の並列計算",
        "category": "simd",
        "optimization_prompt": """Implement an optimized prefix sum (cumulative sum) in C.
Function signature: void prefix_sum(int* arr, int n)

Compute prefix sum in-place: arr[i] = sum of arr[0..i]. Use optimization:
- Break into blocks for better cache usage
- Use temporary storage to reduce dependencies
- Minimize sequential dependencies

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void prefix_sum(int* arr, int n);

int main() {
    int n = 10000000;
    int* arr = malloc(n * sizeof(int));
    int* original = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        arr[i] = 1;
        original[i] = 1;
    }

    // Warmup
    prefix_sum(arr, n);

    // Restore
    for (int i = 0; i < n; i++) arr[i] = original[i];

    // Measure
    clock_t start = clock();
    prefix_sum(arr, n);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Verify (arr[i] should be i+1)
    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (arr[i] != i + 1) {
            correct = 0;
            break;
        }
    }

    free(arr);
    free(original);

    if (correct) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 100.0  # ms
    }
]

# 全10問題を結合
EXTENDED_OPTIMIZATION_PROBLEMS = SIMPLE_OPTIMIZATION_PROBLEMS + NEW_OPTIMIZATION_PROBLEMS

# カテゴリ別の問題数を表示
def print_problem_categories():
    categories = {}
    for problem in EXTENDED_OPTIMIZATION_PROBLEMS:
        category = problem.get('category', 'loop')
        if category not in categories:
            categories[category] = []
        categories[category].append(problem['name'])

    print("Problem Categories:")
    for category, problems in sorted(categories.items()):
        print(f"  {category}: {len(problems)} problems")
        for p in problems:
            print(f"    - {p}")

if __name__ == "__main__":
    print(f"Total problems: {len(EXTENDED_OPTIMIZATION_PROBLEMS)}")
    print_problem_categories()
