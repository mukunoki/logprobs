"""Additional benchmarks designed to stress local-token uncertainty.

These problems are intentionally centered on local indexing, boundary handling,
and reset/update rules.  They are meant to test whether Tail-CBBA helps when
incorrect candidates often contain locally suspicious implementation choices.
"""

PAPER_TAILCASE_PROBLEMS = [
    {
        "name": "stencil2d_halo5",
        "description": "2D 5-point stencil with boundary copy",
        "category": "local_indexing",
        "optimization_prompt": """Implement an optimized 2D 5-point stencil in C.
Function signature:
void stencil2d_halo5(const float* in, float* out, int w, int h, const float coeff[5])

The arrays are row-major with exactly w*h elements.
For boundary cells (x == 0, x == w-1, y == 0, or y == h-1), copy in to out unchanged.
For interior cells:
out[y*w + x] = coeff[0]*center + coeff[1]*left + coeff[2]*right + coeff[3]*up + coeff[4]*down.

Optimize the interior loop order for contiguous x access. Preserve correctness for small w or h.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void stencil2d_halo5(const float* in, float* out, int w, int h, const float coeff[5]);

static float value_at(int i) {
    return (float)(((i * 37 + 11) % 101) - 50) / 17.0f;
}

static int run_case(int w, int h) {
    int n = w * h;
    float* in = (float*)malloc((size_t)n * sizeof(float));
    float* out = (float*)malloc((size_t)n * sizeof(float));
    float* ref = (float*)malloc((size_t)n * sizeof(float));
    if (!in || !out || !ref) return 0;

    const float c[5] = {0.35f, -0.12f, 0.09f, 0.21f, -0.17f};
    for (int i = 0; i < n; ++i) {
        in[i] = value_at(i);
        out[i] = -9999.0f;
        ref[i] = in[i];
    }

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int idx = y * w + x;
            ref[idx] = c[0] * in[idx] + c[1] * in[idx - 1] + c[2] * in[idx + 1]
                     + c[3] * in[idx - w] + c[4] * in[idx + w];
        }
    }

    stencil2d_halo5(in, out, w, h, c);
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (fabsf(out[i] - ref[i]) > 1e-4f) {
            ok = 0;
            break;
        }
    }
    free(in);
    free(out);
    free(ref);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(3, 4) && run_case(17, 19) && run_case(64, 31);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL stencil2d_halo5 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "transpose_strided_f64",
        "description": "rectangular transpose with input/output strides",
        "category": "local_indexing",
        "optimization_prompt": """Implement an optimized rectangular matrix transpose in C.
Function signature:
void transpose_strided_f64(const double* in, double* out, int rows, int cols, int in_stride, int out_stride)

The valid input element at row i, column j is in[i*in_stride + j].
The output matrix has shape cols x rows, and the valid output element is out[j*out_stride + i].
Only write valid transposed elements. Do not modify padding elements in out.
Assume in_stride >= cols and out_stride >= rows.

Use cache-friendly loops where appropriate, but preserve the exact strided indexing.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void transpose_strided_f64(const double* in, double* out, int rows, int cols, int in_stride, int out_stride);

static double val(int i, int j) {
    return (double)((i + 1) * 1000 + (j + 7)) / 13.0;
}

static int run_case(int rows, int cols, int in_stride, int out_stride) {
    double* in = (double*)malloc((size_t)rows * in_stride * sizeof(double));
    double* out = (double*)malloc((size_t)cols * out_stride * sizeof(double));
    if (!in || !out) return 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < in_stride; ++j) {
            in[i * in_stride + j] = (j < cols) ? val(i, j) : -12345.0;
        }
    }
    for (int i = 0; i < cols * out_stride; ++i) out[i] = -777.0;

    transpose_strided_f64(in, out, rows, cols, in_stride, out_stride);

    int ok = 1;
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            if (fabs(out[j * out_stride + i] - val(i, j)) > 1e-12) ok = 0;
        }
        for (int p = rows; p < out_stride; ++p) {
            if (out[j * out_stride + p] != -777.0) ok = 0;
        }
    }
    free(in);
    free(out);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(5, 9, 13, 8) && run_case(17, 11, 19, 23) && run_case(32, 7, 40, 36);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL transpose_strided_f64 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "csr_spmv_alpha_beta",
        "description": "CSR SpMV with alpha and beta update",
        "category": "local_indexing",
        "optimization_prompt": """Implement an optimized CSR sparse matrix-vector multiply in C.
Function signature:
void csr_spmv_alpha_beta(int n, const int* row_ptr, const int* col_idx, const double* values, const double* x, double* y, double alpha, double beta)

For each row i:
sum = values[p] * x[col_idx[p]] for p in row_ptr[i] .. row_ptr[i+1]-1
y[i] = alpha * sum + beta * y[i]

Rows may be empty. Column indices are valid but not necessarily sorted. Preserve the old y[i] value for the beta term.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void csr_spmv_alpha_beta(int n, const int* row_ptr, const int* col_idx, const double* values, const double* x, double* y, double alpha, double beta);

int main(void) {
    const int n = 7;
    const int row_ptr[8] = {0, 3, 3, 5, 9, 10, 10, 13};
    const int col_idx[13] = {0, 4, 2, 1, 6, 3, 0, 5, 3, 2, 6, 1, 4};
    const double values[13] = {1.5, -2.0, 0.25, 3.0, -1.0, 2.5, 0.75, -0.5, 1.25, -3.0, 1.0, 0.5, -1.5};
    double x[7];
    double y[7];
    double ref[7];
    for (int i = 0; i < n; ++i) {
        x[i] = (double)(i + 1) / 3.0;
        y[i] = (double)(10 - i) / 5.0;
        ref[i] = y[i];
    }
    const double alpha = -0.75;
    const double beta = 0.35;

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            sum += values[p] * x[col_idx[p]];
        }
        ref[i] = alpha * sum + beta * ref[i];
    }

    clock_t start = clock();
    for (int iter = 0; iter < 3; ++iter) {
        double yy[7];
        for (int i = 0; i < n; ++i) yy[i] = y[i];
        csr_spmv_alpha_beta(n, row_ptr, col_idx, values, x, yy, alpha, beta);
        for (int i = 0; i < n; ++i) {
            if (fabs(yy[i] - ref[i]) > 1e-10) {
                printf("FAIL csr row %d got %.12f expected %.12f\n", i, yy[i], ref[i]);
                return 1;
            }
        }
    }
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("PASS %.3f\n", ms);
    return 0;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "segmented_prefix_sum_i32",
        "description": "inclusive segmented prefix sum",
        "category": "local_state",
        "optimization_prompt": """Implement an optimized inclusive segmented prefix sum in C.
Function signature:
void segmented_prefix_sum_i32(const int* values, const unsigned char* flags, int* out, int n)

flags[i] == 1 means element i starts a new segment. The output is an inclusive prefix sum within each segment.
For each i, if flags[i] is nonzero, reset the running sum to 0 before adding values[i].
Then store the running sum to out[i].
The first element may or may not have flags[0] == 1. Handle n <= 0.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void segmented_prefix_sum_i32(const int* values, const unsigned char* flags, int* out, int n);

static int run_case(const int* values, const unsigned char* flags, int n) {
    int* out = (int*)malloc((size_t)n * sizeof(int));
    int* ref = (int*)malloc((size_t)n * sizeof(int));
    if (!out || !ref) return 0;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        if (flags[i]) sum = 0;
        sum += values[i];
        ref[i] = sum;
        out[i] = 1234567;
    }
    segmented_prefix_sum_i32(values, flags, out, n);
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (out[i] != ref[i]) {
            ok = 0;
            break;
        }
    }
    free(out);
    free(ref);
    return ok;
}

int main(void) {
    const int n = 23;
    int values[23];
    unsigned char flags[23] = {0};
    for (int i = 0; i < n; ++i) values[i] = (i % 5) - 2;
    flags[0] = 1;
    flags[3] = 1;
    flags[4] = 1;
    flags[11] = 1;
    flags[17] = 1;
    flags[22] = 1;

    int values2[8] = {5, -1, 2, 7, -3, 4, 1, -8};
    unsigned char flags2[8] = {0, 0, 1, 0, 0, 1, 1, 0};

    clock_t start = clock();
    int ok = run_case(values, flags, n) && run_case(values2, flags2, 8);
    int dummy_out = 99;
    segmented_prefix_sum_i32(values, flags, &dummy_out, 0);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL segmented_prefix_sum_i32 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "crop2d_strided_u8",
        "description": "strided 2D crop with untouched output padding",
        "category": "stride_indexing",
        "optimization_prompt": """Implement an optimized strided 2D crop in C.
Function signature:
void crop2d_strided_u8(const unsigned char* src, unsigned char* dst, int src_w, int src_h, int src_stride, int x0, int y0, int crop_w, int crop_h, int dst_stride)

Copy the rectangle src[y0 + y][x0 + x] into dst[y][x] for 0 <= y < crop_h and 0 <= x < crop_w.
Source rows use src_stride bytes. Destination rows use dst_stride bytes.
Assume the crop rectangle is inside the source image, src_stride >= src_w, and dst_stride >= crop_w.
Only write the crop_w valid bytes in each destination row. Do not modify destination padding bytes.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void crop2d_strided_u8(const unsigned char* src, unsigned char* dst, int src_w, int src_h, int src_stride, int x0, int y0, int crop_w, int crop_h, int dst_stride);

static unsigned char pix(int y, int x) {
    return (unsigned char)((y * 29 + x * 17 + 13) & 255);
}

static int run_case(int src_w, int src_h, int src_stride, int x0, int y0, int crop_w, int crop_h, int dst_stride) {
    unsigned char* src = (unsigned char*)malloc((size_t)src_h * src_stride);
    unsigned char* dst = (unsigned char*)malloc((size_t)crop_h * dst_stride);
    if (!src || !dst) return 0;

    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_stride; ++x) {
            src[y * src_stride + x] = (x < src_w) ? pix(y, x) : 0xA5;
        }
    }
    for (int i = 0; i < crop_h * dst_stride; ++i) dst[i] = 0xCC;

    crop2d_strided_u8(src, dst, src_w, src_h, src_stride, x0, y0, crop_w, crop_h, dst_stride);

    int ok = 1;
    for (int y = 0; y < crop_h; ++y) {
        for (int x = 0; x < crop_w; ++x) {
            if (dst[y * dst_stride + x] != pix(y0 + y, x0 + x)) ok = 0;
        }
        for (int x = crop_w; x < dst_stride; ++x) {
            if (dst[y * dst_stride + x] != 0xCC) ok = 0;
        }
    }
    free(src);
    free(dst);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(19, 17, 24, 3, 4, 9, 7, 13)
          && run_case(64, 33, 80, 11, 5, 21, 19, 32)
          && run_case(8, 8, 11, 0, 0, 8, 8, 10);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL crop2d_strided_u8 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "gather_rows_strided_f64",
        "description": "gather selected strided rows into padded output",
        "category": "stride_indexing",
        "optimization_prompt": """Implement an optimized row gather for a strided double matrix in C.
Function signature:
void gather_rows_strided_f64(const double* src, double* dst, const int* rows, int m, int cols, int src_stride, int dst_stride)

The source matrix has at least max(rows)+1 rows and cols valid columns per row.
For each output row r, copy source row rows[r] into destination row r:
dst[r*dst_stride + c] = src[rows[r]*src_stride + c] for 0 <= c < cols.
Source and destination rows may have padding. Assume src_stride >= cols and dst_stride >= cols.
Only write the first cols values of each destination row. Do not modify destination padding.
Rows may contain repeated source indices.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void gather_rows_strided_f64(const double* src, double* dst, const int* rows, int m, int cols, int src_stride, int dst_stride);

static double val(int r, int c) {
    return (double)(r * 101 + c * 7 - 31) / 9.0;
}

static int run_case(int src_rows, int cols, int src_stride, const int* rows, int m, int dst_stride) {
    double* src = (double*)malloc((size_t)src_rows * src_stride * sizeof(double));
    double* dst = (double*)malloc((size_t)m * dst_stride * sizeof(double));
    if (!src || !dst) return 0;

    for (int r = 0; r < src_rows; ++r) {
        for (int c = 0; c < src_stride; ++c) {
            src[r * src_stride + c] = (c < cols) ? val(r, c) : -123456.0;
        }
    }
    for (int i = 0; i < m * dst_stride; ++i) dst[i] = -777.0;

    gather_rows_strided_f64(src, dst, rows, m, cols, src_stride, dst_stride);

    int ok = 1;
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (fabs(dst[r * dst_stride + c] - val(rows[r], c)) > 1e-12) ok = 0;
        }
        for (int c = cols; c < dst_stride; ++c) {
            if (dst[r * dst_stride + c] != -777.0) ok = 0;
        }
    }
    free(src);
    free(dst);
    return ok;
}

int main(void) {
    const int rows1[6] = {4, 1, 7, 1, 0, 5};
    const int rows2[5] = {9, 3, 3, 8, 2};
    clock_t start = clock();
    int ok = run_case(10, 11, 16, rows1, 6, 14)
          && run_case(12, 5, 9, rows2, 5, 8);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL gather_rows_strided_f64 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "scatter_cols_strided_f32",
        "description": "scatter packed columns into selected strided output columns",
        "category": "stride_indexing",
        "optimization_prompt": """Implement an optimized column scatter for strided float matrices in C.
Function signature:
void scatter_cols_strided_f32(const float* src, float* dst, const int* cols, int rows, int k, int src_stride, int dst_stride)

The source matrix has rows x k valid values stored with row stride src_stride.
The destination matrix has rows rows and row stride dst_stride.
For each row r and packed column j:
dst[r*dst_stride + cols[j]] = src[r*src_stride + j].
Only write the selected destination columns. Do not modify any other destination elements.
Assume src_stride >= k and all cols[j] are valid destination column indices.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void scatter_cols_strided_f32(const float* src, float* dst, const int* cols, int rows, int k, int src_stride, int dst_stride);

static float val(int r, int c) {
    return (float)((r * 19 + c * 23 - 17) % 97) / 11.0f;
}

static int is_selected(int c, const int* cols, int k) {
    for (int j = 0; j < k; ++j) if (cols[j] == c) return 1;
    return 0;
}

static int run_case(int rows, int dst_cols, int k, int src_stride, int dst_stride, const int* cols) {
    float* src = (float*)malloc((size_t)rows * src_stride * sizeof(float));
    float* dst = (float*)malloc((size_t)rows * dst_stride * sizeof(float));
    if (!src || !dst) return 0;

    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < src_stride; ++j) {
            src[r * src_stride + j] = (j < k) ? val(r, j) : -1234.0f;
        }
        for (int c = 0; c < dst_stride; ++c) {
            dst[r * dst_stride + c] = -77.0f;
        }
    }

    scatter_cols_strided_f32(src, dst, cols, rows, k, src_stride, dst_stride);

    int ok = 1;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < dst_stride; ++c) {
            if (c < dst_cols && is_selected(c, cols, k)) {
                int packed = -1;
                for (int j = 0; j < k; ++j) if (cols[j] == c) packed = j;
                if (fabsf(dst[r * dst_stride + c] - val(r, packed)) > 1e-6f) ok = 0;
            } else {
                if (dst[r * dst_stride + c] != -77.0f) ok = 0;
            }
        }
    }
    free(src);
    free(dst);
    return ok;
}

int main(void) {
    const int cols1[5] = {7, 0, 4, 9, 2};
    const int cols2[3] = {3, 1, 6};
    clock_t start = clock();
    int ok = run_case(13, 10, 5, 8, 12, cols1)
          && run_case(31, 8, 3, 5, 11, cols2);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL scatter_cols_strided_f32 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "gemm_strided_alpha_beta_f64",
        "description": "row-major strided GEMM with alpha/beta and padding preservation",
        "category": "numeric_stride",
        "optimization_prompt": """Implement an optimized row-major matrix multiplication in C.
Function signature:
void gemm_strided_alpha_beta_f64(const double* A, const double* B, double* C, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta)

Matrices are row-major.
A has shape m x k with row stride lda.
B has shape k x n with row stride ldb.
C has shape m x n with row stride ldc.
Compute C[i,j] = alpha * sum_p A[i,p] * B[p,j] + beta * C[i,j].
Assume lda >= k, ldb >= n, and ldc >= n.
Only write the first n valid values of each C row. Do not modify C padding values.
Use double accumulation and preserve numerical accuracy.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void gemm_strided_alpha_beta_f64(const double* A, const double* B, double* C, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta);

static double aval(int i, int p) { return ((i * 17 + p * 13 + 5) % 37 - 18) / 11.0; }
static double bval(int p, int j) { return ((p * 19 + j * 7 + 3) % 41 - 20) / 13.0; }
static double cval(int i, int j) { return ((i * 23 + j * 5 + 1) % 29 - 14) / 17.0; }

static int run_case(int m, int n, int k, int lda, int ldb, int ldc) {
    double* A = (double*)malloc((size_t)m * lda * sizeof(double));
    double* B = (double*)malloc((size_t)k * ldb * sizeof(double));
    double* C = (double*)malloc((size_t)m * ldc * sizeof(double));
    double* R = (double*)malloc((size_t)m * n * sizeof(double));
    if (!A || !B || !C || !R) return 0;

    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < lda; ++p) A[i * lda + p] = (p < k) ? aval(i, p) : -1234.0;
    }
    for (int p = 0; p < k; ++p) {
        for (int j = 0; j < ldb; ++j) B[p * ldb + j] = (j < n) ? bval(p, j) : -2345.0;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < ldc; ++j) C[i * ldc + j] = (j < n) ? cval(i, j) : -777.0;
    }

    const double alpha = -0.75;
    const double beta = 0.35;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) sum += aval(i, p) * bval(p, j);
            R[i * n + j] = alpha * sum + beta * cval(i, j);
        }
    }

    gemm_strided_alpha_beta_f64(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta);

    int ok = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fabs(C[i * ldc + j] - R[i * n + j]) > 1e-10) ok = 0;
        }
        for (int j = n; j < ldc; ++j) {
            if (C[i * ldc + j] != -777.0) ok = 0;
        }
    }
    free(A); free(B); free(C); free(R);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(3, 5, 4, 7, 8, 9)
          && run_case(8, 6, 7, 11, 10, 13);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL gemm_strided_alpha_beta_f64 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "lower_tri_solve_strided_f64",
        "description": "multiple-RHS lower triangular solve with row strides",
        "category": "numeric_solver",
        "optimization_prompt": """Implement a numerically accurate lower triangular solve in C.
Function signature:
void lower_tri_solve_strided_f64(const double* L, const double* B, double* X, int n, int nrhs, int lda, int ldb, int ldx)

L is a row-major n x n lower triangular matrix with row stride lda. Its diagonal is nonzero and not necessarily 1.
B is a row-major n x nrhs right-hand-side matrix with row stride ldb.
X is a row-major n x nrhs output matrix with row stride ldx.
Solve L * X = B by forward substitution.
Assume lda >= n, ldb >= nrhs, and ldx >= nrhs.
Only write the first nrhs valid values of each X row. Do not modify X padding values.
Use double arithmetic and divide by the diagonal L[i*lda + i].

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void lower_tri_solve_strided_f64(const double* L, const double* B, double* X, int n, int nrhs, int lda, int ldb, int ldx);

static double lval(int i, int j) {
    if (j > i) return 0.0;
    if (i == j) return 2.0 + 0.25 * (double)(i + 1);
    return ((i * 11 + j * 7 + 3) % 17 - 8) / 19.0;
}

static double xtrue(int i, int r) {
    return ((i * 13 + r * 5 + 2) % 23 - 11) / 7.0;
}

static int run_case(int n, int nrhs, int lda, int ldb, int ldx) {
    double* L = (double*)malloc((size_t)n * lda * sizeof(double));
    double* B = (double*)malloc((size_t)n * ldb * sizeof(double));
    double* X = (double*)malloc((size_t)n * ldx * sizeof(double));
    if (!L || !B || !X) return 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < lda; ++j) L[i * lda + j] = (j < n) ? lval(i, j) : -1234.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < ldb; ++r) B[i * ldb + r] = -2345.0;
        for (int r = 0; r < nrhs; ++r) {
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) sum += lval(i, j) * xtrue(j, r);
            B[i * ldb + r] = sum;
        }
        for (int r = 0; r < ldx; ++r) X[i * ldx + r] = -777.0;
    }

    lower_tri_solve_strided_f64(L, B, X, n, nrhs, lda, ldb, ldx);

    int ok = 1;
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < nrhs; ++r) {
            if (fabs(X[i * ldx + r] - xtrue(i, r)) > 1e-9) ok = 0;
        }
        for (int r = nrhs; r < ldx; ++r) {
            if (X[i * ldx + r] != -777.0) ok = 0;
        }
    }
    free(L); free(B); free(X);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(5, 3, 8, 6, 7)
          && run_case(11, 2, 13, 5, 4);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL lower_tri_solve_strided_f64 mismatch\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
]
