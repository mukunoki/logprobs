"""
Simple Performance Optimization Problems
より簡単な性能最適化タスク（基本的な最適化のみ）
"""

SIMPLE_OPTIMIZATION_PROBLEMS = [
    {
        "name": "array_sum_unroll",
        "description": "配列総和（ループアンローリング）",
        "category": "loop",
        "optimization_prompt": """Implement an optimized array sum in C.
Function signature: double sum_array(double* arr, int n)

Use loop unrolling by 4 to improve performance. Example structure:
- Process 4 elements per iteration
- Handle remaining elements separately
- Use separate accumulators for each unrolled iteration

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double sum_array(double* arr, int n);

int main() {
    int n = 10000000;
    double* arr = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        arr[i] = 1.0;
    }

    // Warmup
    double result = sum_array(arr, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        result = sum_array(arr, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    free(arr);

    if (fabs(result - n) < 1.0) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%.0f expected=%d\\n", result, n);
        return 1;
    }
}
""",
        "performance_threshold": 20.0  # ms
    },
    {
        "name": "dot_product_unroll",
        "description": "内積計算（ループアンローリング）",
        "category": "loop",
        "optimization_prompt": """Implement an optimized dot product in C.
Function signature: double dot_product(double* a, double* b, int n)

Use loop unrolling by 4 to improve performance. Example structure:
- Process 4 pairs per iteration
- Use 4 separate accumulators
- Handle remaining elements separately

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double dot_product(double* a, double* b, int n);

int main() {
    int n = 10000000;
    double* a = malloc(n * sizeof(double));
    double* b = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    // Warmup
    double result = dot_product(a, b, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        result = dot_product(a, b, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    free(a);
    free(b);

    double expected = 2.0 * n;
    if (fabs(result - expected) < 1.0) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%.0f expected=%.0f\\n", result, expected);
        return 1;
    }
}
""",
        "performance_threshold": 20.0  # ms
    },
    {
        "name": "max_value_branchless",
        "description": "最大値探索（分岐削減）",
        "category": "branch",
        "optimization_prompt": """Implement an optimized maximum value finder in C.
Function signature: double find_max(double* arr, int n)

Use branchless techniques to reduce branch misprediction penalty:
- Use conditional move instead of if statements
- Example: max = (arr[i] > max) ? arr[i] : max; is better than if (arr[i] > max) max = arr[i];

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double find_max(double* arr, int n);

int main() {
    int n = 10000000;
    double* arr = malloc(n * sizeof(double));

    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = (double)(rand() % 1000);
    }
    arr[n/2] = 9999.0;  // Known maximum

    // Warmup
    double result = find_max(arr, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        result = find_max(arr, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    free(arr);

    if (result == 9999.0) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%.0f expected=9999\\n", result);
        return 1;
    }
}
""",
        "performance_threshold": 25.0  # ms
    },
    {
        "name": "string_copy_bulk",
        "description": "文字列コピー（バルクコピー）",
        "category": "loop",
        "optimization_prompt": """Implement an optimized string copy in C.
Function signature: void string_copy(char* dest, const char* src, int n)

Copy n characters from src to dest. Use bulk copy optimization:
- Copy multiple bytes at once (e.g., 8 bytes using long*)
- Handle alignment properly
- Process remaining bytes individually

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void string_copy(char* dest, const char* src, int n);

int main() {
    int n = 10000000;
    char* src = malloc(n);
    char* dest = malloc(n);

    for (int i = 0; i < n; i++) {
        src[i] = 'a' + (i % 26);
    }

    // Warmup
    string_copy(dest, src, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        string_copy(dest, src, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    // Verify
    int correct = 1;
    for (int i = 0; i < n; i++) {
        if (dest[i] != src[i]) {
            correct = 0;
            break;
        }
    }

    free(src);
    free(dest);

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
        "name": "count_positive_cmov",
        "description": "正の値のカウント（条件移動命令）",
        "category": "branch",
        "optimization_prompt": """Implement an optimized function to count positive values in C.
Function signature: int count_positive(int* arr, int n)

Count how many positive values (> 0) are in the array. Use branchless techniques:
- Use conditional move: count += (arr[i] > 0) ? 1 : 0;
- This avoids branch misprediction
- Can also use: count += (arr[i] > 0);

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int count_positive(int* arr, int n);

int main() {
    int n = 10000000;
    int* arr = malloc(n * sizeof(int));

    srand(42);
    int expected = 0;
    for (int i = 0; i < n; i++) {
        arr[i] = (rand() % 200) - 100;  // Range: -100 to 99
        if (arr[i] > 0) expected++;
    }

    // Warmup
    int result = count_positive(arr, n);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        result = count_positive(arr, n);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    free(arr);

    if (result == expected) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%d expected=%d\\n", result, expected);
        return 1;
    }
}
""",
        "performance_threshold": 30.0  # ms
    }
]
