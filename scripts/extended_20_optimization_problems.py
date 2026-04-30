"""
Extended 20 Performance Optimization Problems

Simple 5 + Extended 10からさらに10問追加し、合計20問のベンチマーク

新規追加10問のカテゴリ:
- メモリ最適化（2問）
- データ構造最適化（3問）
- アルゴリズム（3問）
- 並列化（1問）
- ビット操作（1問）
"""

# 既存のExtended 10問をインポート
from extended_optimization_problems import EXTENDED_OPTIMIZATION_PROBLEMS

# 新規10問を追加
ADDITIONAL_10_PROBLEMS = [
    # メモリ最適化カテゴリ
    {
        "name": "struct_packing",
        "description": "構造体パッキング最適化",
        "category": "memory",
        "optimization_prompt": """Implement an optimized structure packing in C to minimize memory usage.

Define a struct Person with fields:
- char name[32]
- int age
- char gender (single character)
- double salary

Reorder fields to minimize padding and memory footprint. Implement:
void process_people(struct Person* people, int n, double* avg_salary)

Calculate average salary efficiently with good cache locality.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Person {
    char name[32];
    int age;
    char gender;
    double salary;
} __attribute__((packed));

void process_people(struct Person* people, int n, double* avg_salary);

int main() {
    int n = 1000000;
    struct Person* people = malloc(n * sizeof(struct Person));

    for (int i = 0; i < n; i++) {
        snprintf(people[i].name, 32, "Person%d", i);
        people[i].age = 20 + (i % 60);
        people[i].gender = (i % 2) ? 'M' : 'F';
        people[i].salary = 30000.0 + (i % 50000);
    }

    double avg_salary;

    // Warmup
    process_people(people, n, &avg_salary);

    // Measure
    clock_t start = clock();
    for (int iter = 0; iter < 10; iter++) {
        process_people(people, n, &avg_salary);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0 / 10.0;

    free(people);

    // Expected average around 54999.5
    if (avg_salary > 50000 && avg_salary < 60000) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL avg_salary=%.2f\\n", avg_salary);
        return 1;
    }
}
""",
        "performance_threshold": 50.0  # ms
    },
    {
        "name": "memory_pool_allocation",
        "description": "メモリプール最適化",
        "category": "memory",
        "optimization_prompt": """Implement an optimized memory allocation using a simple memory pool in C.

Function signatures:
void* pool_alloc(size_t size)
void pool_free(void* ptr)
void pool_init(size_t total_size)

Use a pre-allocated memory pool to avoid repeated malloc/free calls.
Implement simple bump-pointer allocation with optional free list.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void* pool_alloc(size_t size);
void pool_free(void* ptr);
void pool_init(size_t total_size);

int main() {
    pool_init(100000000);  // 100MB pool

    int n = 1000000;
    void** ptrs = malloc(n * sizeof(void*));

    // Warmup
    for (int i = 0; i < 1000; i++) {
        ptrs[i] = pool_alloc(64);
    }

    // Measure
    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        ptrs[i] = pool_alloc(64);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(ptrs);

    printf("PASS %.2f\\n", time_ms);
    return 0;
}
""",
        "performance_threshold": 100.0  # ms
    },

    # データ構造最適化カテゴリ
    {
        "name": "hash_table_optimization",
        "description": "ハッシュテーブル最適化",
        "category": "data_structure",
        "optimization_prompt": """Implement an optimized hash table in C with chaining.

Function signatures:
void hash_insert(int key, int value)
int hash_lookup(int key)
void hash_init(int capacity)

Use efficient hash function (e.g., multiplicative hashing).
Optimize cache locality by using array-based chaining.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void hash_insert(int key, int value);
int hash_lookup(int key);
void hash_init(int capacity);

int main() {
    hash_init(10000);

    int n = 100000;

    // Insert
    for (int i = 0; i < n; i++) {
        hash_insert(i, i * 2);
    }

    // Warmup lookup
    for (int i = 0; i < 1000; i++) {
        hash_lookup(i);
    }

    // Measure lookup
    clock_t start = clock();
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += hash_lookup(i);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Expected sum: 0*2 + 1*2 + ... + (n-1)*2 = 2 * (n-1)*n/2 = n*(n-1)
    long long expected = (long long)n * (n - 1);
    if (sum == expected) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL sum=%d expected=%lld\\n", sum, expected);
        return 1;
    }
}
""",
        "performance_threshold": 50.0  # ms
    },
    {
        "name": "linked_list_to_array",
        "description": "リンクリスト→配列変換",
        "category": "data_structure",
        "optimization_prompt": """Implement an optimized array-based list instead of linked list in C.

Function signatures:
void list_init(int capacity)
void list_append(int value)
int list_get(int index)
int list_size()

Use dynamic array with amortized O(1) append (doubling strategy).
Optimize for cache locality compared to linked list.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void list_init(int capacity);
void list_append(int value);
int list_get(int index);
int list_size();

int main() {
    list_init(1000);

    int n = 1000000;

    // Warmup
    for (int i = 0; i < 1000; i++) {
        list_append(i);
    }

    // Measure append
    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        list_append(i);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Verify
    int size = list_size();
    int val = list_get(n - 1);

    if (size >= n && val == n - 1) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL size=%d val=%d\\n", size, val);
        return 1;
    }
}
""",
        "performance_threshold": 100.0  # ms
    },
    {
        "name": "binary_heap_implementation",
        "description": "バイナリヒープ実装",
        "category": "data_structure",
        "optimization_prompt": """Implement an optimized binary min-heap in C.

Function signatures:
void heap_init(int capacity)
void heap_insert(int value)
int heap_extract_min()
int heap_size()

Use array-based binary heap with efficient heapify operations.
Optimize for cache locality with compact representation.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void heap_init(int capacity);
void heap_insert(int value);
int heap_extract_min();
int heap_size();

int main() {
    heap_init(100000);

    int n = 100000;

    // Insert random values
    srand(42);
    for (int i = 0; i < n; i++) {
        heap_insert(rand() % 100000);
    }

    // Warmup
    for (int i = 0; i < 10; i++) {
        heap_extract_min();
    }

    // Measure extract
    clock_t start = clock();
    int prev = -1;
    int sorted = 1;
    for (int i = 0; i < 1000; i++) {
        int val = heap_extract_min();
        if (val < prev) sorted = 0;
        prev = val;
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    if (sorted) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 10.0  # ms
    },

    # アルゴリズムカテゴリ
    {
        "name": "merge_sort_optimization",
        "description": "マージソート最適化",
        "category": "algorithm",
        "optimization_prompt": """Implement an optimized merge sort in C.

Function signature: void merge_sort(int* arr, int n)

Use optimizations:
- In-place merge or minimal extra memory
- Switch to insertion sort for small subarrays (threshold ~16)
- Iterative bottom-up approach for better cache locality

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge_sort(int* arr, int n);

int main() {
    int n = 100000;
    int* arr = malloc(n * sizeof(int));
    int* original = malloc(n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100000;
        original[i] = arr[i];
    }

    // Warmup
    merge_sort(arr, n);

    // Restore
    for (int i = 0; i < n; i++) arr[i] = original[i];

    // Measure
    clock_t start = clock();
    merge_sort(arr, n);
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
        "performance_threshold": 100.0  # ms
    },
    {
        "name": "heap_sort_implementation",
        "description": "ヒープソート実装",
        "category": "algorithm",
        "optimization_prompt": """Implement an optimized heap sort in C.

Function signature: void heap_sort(int* arr, int n)

Use in-place heapify and extract operations.
Optimize heapify for cache locality.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void heap_sort(int* arr, int n);

int main() {
    int n = 100000;
    int* arr = malloc(n * sizeof(int));
    int* original = malloc(n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100000;
        original[i] = arr[i];
    }

    // Warmup
    heap_sort(arr, n);

    // Restore
    for (int i = 0; i < n; i++) arr[i] = original[i];

    // Measure
    clock_t start = clock();
    heap_sort(arr, n);
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
        "performance_threshold": 150.0  # ms
    },
    {
        "name": "string_search_kmp",
        "description": "文字列検索（KMP法）",
        "category": "algorithm",
        "optimization_prompt": """Implement an optimized string search in C using KMP algorithm.

Function signature: int string_search(const char* text, const char* pattern)

Use Knuth-Morris-Pratt algorithm:
- Build failure function efficiently
- Linear time search O(n+m)
- Return index of first occurrence, -1 if not found

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int string_search(const char* text, const char* pattern);

int main() {
    // Create large text
    int text_len = 10000000;
    char* text = malloc(text_len + 1);
    for (int i = 0; i < text_len; i++) {
        text[i] = 'a' + (i % 26);
    }
    text[text_len] = '\\0';

    // Insert pattern at known position
    const char* pattern = "abcdefghij";
    int pattern_len = strlen(pattern);
    int known_pos = text_len / 2;
    memcpy(text + known_pos, pattern, pattern_len);

    // Warmup
    string_search(text, pattern);

    // Measure
    clock_t start = clock();
    int result = string_search(text, pattern);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(text);

    if (result == known_pos) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL result=%d expected=%d\\n", result, known_pos);
        return 1;
    }
}
""",
        "performance_threshold": 100.0  # ms
    },

    # 並列化カテゴリ
    {
        "name": "matrix_multiply_blocked",
        "description": "ブロック行列積",
        "category": "parallel",
        "optimization_prompt": """Implement an optimized blocked matrix multiplication in C.

Function signature: void matrix_multiply(int n, double A[n][n], double B[n][n], double C[n][n])

Compute C = A * B using cache blocking:
- Tile/block matrices (e.g., 32x32 or 64x64 blocks)
- Process blocks to maximize cache reuse
- Use loop interchange for better locality

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void matrix_multiply(int n, double A[n][n], double B[n][n], double C[n][n]);

int main() {
    int n = 256;
    double (*A)[n] = malloc(sizeof(double[n][n]));
    double (*B)[n] = malloc(sizeof(double[n][n]));
    double (*C)[n] = malloc(sizeof(double[n][n]));

    // Initialize
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
    }

    // Warmup
    matrix_multiply(n, A, B, C);

    // Measure
    clock_t start = clock();
    matrix_multiply(n, A, B, C);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Verify (each element should be n)
    int correct = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C[i][j] - n) > 0.01) {
                correct = 0;
                goto done;
            }
        }
    }
done:
    free(A);
    free(B);
    free(C);

    if (correct) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL\\n");
        return 1;
    }
}
""",
        "performance_threshold": 200.0  # ms
    },

    # ビット操作カテゴリ
    {
        "name": "popcount_optimization",
        "description": "ビットカウント最適化",
        "category": "bitwise",
        "optimization_prompt": """Implement an optimized population count (count 1 bits) in C.

Function signature: int popcount(unsigned int x)

Use bit manipulation tricks:
- Brian Kernighan's algorithm or
- Parallel bit counting or
- Lookup table

Return the number of 1 bits in x.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int popcount(unsigned int x);

int main() {
    int n = 100000000;
    unsigned int* values = malloc(n * sizeof(unsigned int));

    srand(42);
    for (int i = 0; i < n; i++) {
        values[i] = rand();
    }

    // Warmup
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += popcount(values[i]);
    }

    // Measure
    clock_t start = clock();
    sum = 0;
    for (int i = 0; i < n; i++) {
        sum += popcount(values[i]);
    }
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(values);

    // Expected: around 16 bits per 32-bit value on average
    if (sum > n * 10 && sum < n * 22) {
        printf("PASS %.2f\\n", time_ms);
        return 0;
    } else {
        printf("FAIL sum=%d\\n", sum);
        return 1;
    }
}
""",
        "performance_threshold": 500.0  # ms
    }
]

# 全20問題を結合
EXTENDED_20_OPTIMIZATION_PROBLEMS = EXTENDED_OPTIMIZATION_PROBLEMS + ADDITIONAL_10_PROBLEMS

# カテゴリ別の問題数を表示
def print_problem_categories():
    categories = {}
    for problem in EXTENDED_20_OPTIMIZATION_PROBLEMS:
        category = problem.get('category', 'loop')
        if category not in categories:
            categories[category] = []
        categories[category].append(problem['name'])

    print("Problem Categories (Extended 20):")
    for category, problems in sorted(categories.items()):
        print(f"  {category}: {len(problems)} problems")
        for p in problems:
            print(f"    - {p}")

if __name__ == "__main__":
    print(f"Total problems: {len(EXTENDED_20_OPTIMIZATION_PROBLEMS)}")
    print()
    print_problem_categories()
