"""General-purpose hard benchmarks for evaluating Tail-CBBA.

These benchmarks avoid artificial raw-stride tasks and focus on common code:
dynamic programming, numerically stable ML kernels, graph analytics, numerical
linear algebra, and strict byte-level parsing.
"""

PAPER_GENERAL_HARD_PROBLEMS = [
    {
        "name": "banded_edit_distance_i32",
        "description": "edit distance with cutoff and banded dynamic programming",
        "category": "dynamic_programming",
        "optimization_prompt": """Implement an optimized edit distance function in C.
Function signature:
int banded_edit_distance_i32(const unsigned char* a, int n, const unsigned char* b, int m, int max_dist)

Return the Levenshtein edit distance between byte strings a and b if it is <= max_dist.
If the true distance is greater than max_dist, return max_dist + 1.
Handle insertions, deletions, and substitutions with cost 1.
Use a banded or otherwise efficient dynamic-programming implementation when possible.
Handle n == 0, m == 0, max_dist < 0, and length differences larger than max_dist.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int banded_edit_distance_i32(const unsigned char* a, int n, const unsigned char* b, int m, int max_dist);

static int ref_edit(const unsigned char* a, int n, const unsigned char* b, int m) {
    int* prev = (int*)malloc((size_t)(m + 1) * sizeof(int));
    int* cur = (int*)malloc((size_t)(m + 1) * sizeof(int));
    if (!prev || !cur) return 999999;
    for (int j = 0; j <= m; ++j) prev[j] = j;
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        for (int j = 1; j <= m; ++j) {
            int sub = prev[j - 1] + (a[i - 1] != b[j - 1]);
            int del = prev[j] + 1;
            int ins = cur[j - 1] + 1;
            int v = sub < del ? sub : del;
            cur[j] = v < ins ? v : ins;
        }
        int* tmp = prev; prev = cur; cur = tmp;
    }
    int ans = prev[m];
    free(prev);
    free(cur);
    return ans;
}

static int run_case(const char* x, const char* y, int max_dist) {
    int n = (int)strlen(x);
    int m = (int)strlen(y);
    int exact = ref_edit((const unsigned char*)x, n, (const unsigned char*)y, m);
    int expect = exact <= max_dist ? exact : max_dist + 1;
    int got = banded_edit_distance_i32((const unsigned char*)x, n, (const unsigned char*)y, m, max_dist);
    if (got != expect) {
        printf("FAIL edit '%s' '%s' k=%d got %d expected %d exact %d\n", x, y, max_dist, got, expect, exact);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    ok &= run_case("", "", 0);
    ok &= run_case("", "abc", 2);
    ok &= run_case("kitten", "sitting", 3);
    ok &= run_case("kitten", "sitting", 2);
    ok &= run_case("abcdef", "abqdef", 1);
    ok &= run_case("abcdef", "azced", 3);
    ok &= run_case("aaaaaaaaab", "aaaaaaaabb", 2);
    ok &= run_case("01234567890123456789", "01234599990123456789", 3);
    ok &= run_case("long_common_prefix_then_diff_x", "long_common_prefix_then_diff_yz", 2);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "softmax_cross_entropy_stable_f32",
        "description": "numerically stable softmax cross entropy with ignored labels",
        "category": "machine_learning",
        "optimization_prompt": """Implement a numerically stable softmax cross entropy kernel in C.
Function signature:
float softmax_cross_entropy_stable_f32(const float* logits, const int* labels, float* losses, int batch, int classes, int ignore_index)

logits is a row-major batch x classes array.
For each row i:
- If labels[i] == ignore_index, set losses[i] = 0.0f and exclude it from the mean.
- Otherwise labels[i] is a valid class index.
- Compute loss = log(sum_j exp(logits[i,j] - max_j logits[i,j])) + max_j logits[i,j] - logits[i,label].
Return the mean loss over non-ignored rows. If all rows are ignored, return 0.0f.
Use float outputs but accumulate carefully enough for large positive or negative logits.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float softmax_cross_entropy_stable_f32(const float* logits, const int* labels, float* losses, int batch, int classes, int ignore_index);

static double ref_row(const float* row, int classes, int label) {
    double maxv = row[0];
    for (int j = 1; j < classes; ++j) if (row[j] > maxv) maxv = row[j];
    double sum = 0.0;
    for (int j = 0; j < classes; ++j) sum += exp((double)row[j] - maxv);
    return log(sum) + maxv - (double)row[label];
}

static int run_case(const float* logits, const int* labels, int batch, int classes, int ignore_index) {
    float* losses = (float*)malloc((size_t)batch * sizeof(float));
    if (!losses) return 0;
    for (int i = 0; i < batch; ++i) losses[i] = -99.0f;
    float got = softmax_cross_entropy_stable_f32(logits, labels, losses, batch, classes, ignore_index);
    double total = 0.0;
    int count = 0;
    int ok = 1;
    for (int i = 0; i < batch; ++i) {
        double expect = 0.0;
        if (labels[i] != ignore_index) {
            expect = ref_row(logits + (size_t)i * classes, classes, labels[i]);
            total += expect;
            count++;
        }
        if (fabs((double)losses[i] - expect) > 2e-4) ok = 0;
    }
    double mean = count ? total / count : 0.0;
    if (fabs((double)got - mean) > 2e-4) ok = 0;
    if (!ok) printf("FAIL softmax got %.8f expected %.8f\n", got, mean);
    free(losses);
    return ok;
}

int main(void) {
    clock_t start = clock();
    const float logits1[] = {
        1.0f, 2.0f, 3.0f, -4.0f,
        1000.0f, 999.0f, 998.0f, 997.0f,
        -1000.0f, -1001.0f, -999.0f, -1002.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    const int labels1[] = {2, 0, 2, -1};
    const float logits2[] = {
        12.0f, -5.0f, 0.5f,
        -3.0f, 7.0f, 2.0f,
        20.0f, 20.0f, 19.0f
    };
    const int labels2[] = {0, 1, 2};
    const int labels_ignored[] = {-7, -7};
    const float logits3[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int ok = run_case(logits1, labels1, 4, 4, -1)
          && run_case(logits2, labels2, 3, 3, -1)
          && run_case(logits3, labels_ignored, 2, 2, -7);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "csr_pagerank_step_f64",
        "description": "one PageRank iteration on CSR graph with dangling nodes",
        "category": "graph_analytics",
        "optimization_prompt": """Implement one PageRank iteration for a directed graph in C.
Function signature:
void csr_pagerank_step_f64(int n, const int* row_ptr, const int* col_idx, const double* rank, const int* out_degree, double* next, double damping)

The graph is stored as CSR by source vertex: outgoing edges of vertex i are col_idx[row_ptr[i]..row_ptr[i+1]-1].
out_degree[i] is the number of outgoing edges for vertex i.
If out_degree[i] == 0, vertex i is dangling and its rank contributes uniformly to all vertices.
For every vertex v:
next[v] = (1 - damping) / n + damping * (sum of rank[u] / out_degree[u] over edges u->v + dangling_sum / n)
Initialize all next values inside the function. Preserve correctness for n <= 0.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void csr_pagerank_step_f64(int n, const int* row_ptr, const int* col_idx, const double* rank, const int* out_degree, double* next, double damping);

static int run_case(int n, const int* row_ptr, const int* col_idx, const double* rank, double damping) {
    int* out_degree = (int*)malloc((size_t)n * sizeof(int));
    double* next = (double*)malloc((size_t)n * sizeof(double));
    double* ref = (double*)malloc((size_t)n * sizeof(double));
    if (!out_degree || !next || !ref) return 0;
    double dangling = 0.0;
    for (int i = 0; i < n; ++i) {
        out_degree[i] = row_ptr[i + 1] - row_ptr[i];
        next[i] = -1.0;
        ref[i] = (1.0 - damping) / n;
        if (out_degree[i] == 0) dangling += rank[i];
    }
    for (int i = 0; i < n; ++i) ref[i] += damping * dangling / n;
    for (int u = 0; u < n; ++u) {
        if (out_degree[u] == 0) continue;
        double contrib = damping * rank[u] / out_degree[u];
        for (int p = row_ptr[u]; p < row_ptr[u + 1]; ++p) {
            ref[col_idx[p]] += contrib;
        }
    }
    csr_pagerank_step_f64(n, row_ptr, col_idx, rank, out_degree, next, damping);
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (fabs(next[i] - ref[i]) > 1e-12) {
            printf("FAIL pagerank %d got %.15f expected %.15f\n", i, next[i], ref[i]);
            ok = 0;
        }
    }
    free(out_degree);
    free(next);
    free(ref);
    return ok;
}

int main(void) {
    clock_t start = clock();
    const int rp1[] = {0, 2, 3, 3, 5, 6};
    const int ci1[] = {1, 2, 2, 0, 2, 3};
    const double r1[] = {0.20, 0.10, 0.30, 0.25, 0.15};
    const int rp2[] = {0, 0, 2, 3, 5};
    const int ci2[] = {0, 2, 1, 1, 3};
    const double r2[] = {0.4, 0.3, 0.2, 0.1};
    int ok = run_case(5, rp1, ci1, r1, 0.85) && run_case(4, rp2, ci2, r2, 0.70);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "utf8_validate_strict",
        "description": "strict UTF-8 validator rejecting overlong and invalid ranges",
        "category": "byte_parsing",
        "optimization_prompt": """Implement a strict UTF-8 validator in C.
Function signature:
int utf8_validate_strict(const unsigned char* s, int n)

Return 1 if s[0..n-1] is valid UTF-8, otherwise return 0.
Reject truncated sequences, invalid continuation bytes, overlong encodings, UTF-16 surrogate code points, and code points greater than U+10FFFF.
Accept valid 1-byte, 2-byte, 3-byte, and 4-byte sequences.
Handle n < 0 as invalid and n == 0 as valid.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <time.h>

int utf8_validate_strict(const unsigned char* s, int n);

static int check(const unsigned char* s, int n, int expect, const char* name) {
    int got = utf8_validate_strict(s, n);
    if (got != expect) {
        printf("FAIL utf8 %s got %d expected %d\n", name, got, expect);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    const unsigned char ascii[] = "hello";
    const unsigned char valid2[] = {0xC2, 0xA2};
    const unsigned char valid3[] = {0xE2, 0x82, 0xAC};
    const unsigned char valid4[] = {0xF0, 0x9F, 0x98, 0x80};
    const unsigned char mixed[] = {'A', 0xE3, 0x81, 0x82, 'B', 0xC3, 0xA9};
    const unsigned char cont_alone[] = {0x80};
    const unsigned char overlong2[] = {0xC0, 0xAF};
    const unsigned char overlong3[] = {0xE0, 0x80, 0xAF};
    const unsigned char surrogate[] = {0xED, 0xA0, 0x80};
    const unsigned char too_large[] = {0xF4, 0x90, 0x80, 0x80};
    const unsigned char truncated[] = {0xF0, 0x9F, 0x98};
    const unsigned char bad_cont[] = {0xE2, 0x28, 0xA1};
    const unsigned char max_valid[] = {0xF4, 0x8F, 0xBF, 0xBF};
    int ok = 1;
    ok &= check(ascii, 5, 1, "ascii");
    ok &= check(valid2, 2, 1, "valid2");
    ok &= check(valid3, 3, 1, "valid3");
    ok &= check(valid4, 4, 1, "valid4");
    ok &= check(mixed, 7, 1, "mixed");
    ok &= check(max_valid, 4, 1, "max_valid");
    ok &= check(cont_alone, 1, 0, "cont_alone");
    ok &= check(overlong2, 2, 0, "overlong2");
    ok &= check(overlong3, 3, 0, "overlong3");
    ok &= check(surrogate, 3, 0, "surrogate");
    ok &= check(too_large, 4, 0, "too_large");
    ok &= check(truncated, 3, 0, "truncated");
    ok &= check(bad_cont, 3, 0, "bad_cont");
    ok &= check(ascii, -1, 0, "negative");
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "cholesky_spd_f64",
        "description": "Cholesky decomposition with non-SPD rejection",
        "category": "numerical_linear_algebra",
        "optimization_prompt": """Implement a Cholesky decomposition routine in C.
Function signature:
int cholesky_spd_f64(const double* A, double* L, int n)

A is a row-major n x n symmetric matrix.
If A is symmetric positive definite, write a lower-triangular matrix L such that A = L * L^T and return 1.
Set entries above the diagonal of L to 0.0.
If A is not positive definite, return 0.
Handle n <= 0 by returning 0.
Use double precision and avoid modifying A.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int cholesky_spd_f64(const double* A, double* L, int n);

static int check_factor(const double* A, const double* L, int n, double tol, const char* name) {
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (fabs(L[(size_t)i * n + j]) > tol) {
                printf("FAIL cholesky %s upper L[%d,%d]=%.17g\n", name, i, j, L[(size_t)i * n + j]);
                ok = 0;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            int limit = i < j ? i : j;
            for (int k = 0; k <= limit; ++k) {
                s += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            }
            if (fabs(s - A[(size_t)i * n + j]) > tol) {
                printf("FAIL cholesky %s recon[%d,%d] got %.17g expected %.17g\n", name, i, j, s, A[(size_t)i * n + j]);
                ok = 0;
            }
        }
    }
    return ok;
}

static int run_spd(const double* A, int n, double tol, const char* name) {
    double* L = (double*)malloc((size_t)n * n * sizeof(double));
    double* A_copy = (double*)malloc((size_t)n * n * sizeof(double));
    if (!L || !A_copy) return 0;
    for (int i = 0; i < n * n; ++i) {
        L[i] = 12345.0;
        A_copy[i] = A[i];
    }
    int got = cholesky_spd_f64(A, L, n);
    int ok = 1;
    if (got != 1) {
        printf("FAIL cholesky %s returned %d for SPD matrix\n", name, got);
        ok = 0;
    } else {
        ok &= check_factor(A, L, n, tol, name);
    }
    for (int i = 0; i < n * n; ++i) {
        if (A[i] != A_copy[i]) {
            printf("FAIL cholesky %s modified A at %d\n", name, i);
            ok = 0;
            break;
        }
    }
    free(L);
    free(A_copy);
    return ok;
}

static int run_not_spd(const double* A, int n, const char* name) {
    double* L = (double*)malloc((size_t)n * n * sizeof(double));
    if (!L) return 0;
    for (int i = 0; i < n * n; ++i) L[i] = -7.0;
    int got = cholesky_spd_f64(A, L, n);
    free(L);
    if (got != 0) {
        printf("FAIL cholesky %s returned %d for non-SPD matrix\n", name, got);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    const double spd3[] = {
        25.0, 15.0, -5.0,
        15.0, 18.0,  0.0,
        -5.0,  0.0, 11.0
    };
    const double spd4[] = {
        6.00,  2.00,  1.00,  0.50,
        2.00,  5.00, -1.00,  1.50,
        1.00, -1.00,  4.50, -0.75,
        0.50,  1.50, -0.75, 3.25
    };
    const double near_spd[] = {
        1.000001, 0.999999,
        0.999999, 1.000001
    };
    const double not_spd_zero_pivot[] = {
        1.0, 2.0,
        2.0, 4.0
    };
    const double not_spd_negative[] = {
        2.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, 3.0
    };
    int ok = 1;
    ok &= run_spd(spd3, 3, 1e-9, "spd3");
    ok &= run_spd(spd4, 4, 1e-9, "spd4");
    ok &= run_spd(near_spd, 2, 1e-9, "near_spd");
    ok &= run_not_spd(not_spd_zero_pivot, 2, "zero_pivot");
    ok &= run_not_spd(not_spd_negative, 3, "negative_diag");
    if (cholesky_spd_f64(spd3, (double*)spd3, 0) != 0) {
        printf("FAIL cholesky n0\n");
        ok = 0;
    }
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
]
