"""Paper numerical source problems for CBBA.

The tasks combine performance-oriented C kernels with numerical accuracy
requirements.  Each problem asks the model to emit only function
implementations; the test harness checks correctness and reports runtime.
"""

PAPER_SOURCE_NUMERICAL_PROBLEMS = [
    {
        "name": "matmul_mixed_blocked",
        "description": "Mixed-precision blocked matrix multiplication",
        "category": "mixed_precision",
        "optimization_prompt": """Implement an optimized mixed-precision matrix multiplication in C.

Function signature:
void matmul_mixed_blocked(const float* A, const float* B, float* C, int M, int N, int K)

Compute C = A x B for row-major matrices:
- A is M x K, stored as float
- B is K x N, stored as float
- C is M x N, stored as float

Requirements:
- Use cache blocking, loop tiling, or a cache-friendly loop order.
- Use float inputs but accumulate each dot product in double or compensated float accumulation.
- Return results in float.
- Do not allocate large temporary matrices.
- Correctly handle dimensions that are not multiples of the block size.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void matmul_mixed_blocked(const float* A, const float* B, float* C, int M, int N, int K);

static float val_a(int i, int k) {
    return (float)(((i * 17 + k * 13) % 29) - 14) / 11.0f;
}

static float val_b(int k, int j) {
    return (float)(((k * 7 - j * 19) % 31) - 15) / 13.0f;
}

static int run_case(int M, int N, int K) {
    float* A = (float*)malloc((size_t)M * K * sizeof(float));
    float* B = (float*)malloc((size_t)K * N * sizeof(float));
    float* C = (float*)malloc((size_t)M * N * sizeof(float));
    double* R = (double*)malloc((size_t)M * N * sizeof(double));
    if (!A || !B || !C || !R) return 0;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) A[i * K + k] = val_a(i, k);
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) B[k * N + j] = val_b(k, j);
    }
    for (int i = 0; i < M * N; i++) C[i] = -9999.0f;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) sum += (double)A[i * K + k] * (double)B[k * N + j];
            R[i * N + j] = sum;
        }
    }

    matmul_mixed_blocked(A, B, C, M, N, K);

    double max_abs = 0.0;
    double ref_norm = 0.0;
    double err_norm = 0.0;
    for (int i = 0; i < M * N; i++) {
        double err = fabs((double)C[i] - R[i]);
        if (err > max_abs) max_abs = err;
        err_norm += err * err;
        ref_norm += R[i] * R[i];
    }

    free(A); free(B); free(C); free(R);
    double rel = sqrt(err_norm / (ref_norm + 1e-30));
    return max_abs < 2e-3 && rel < 1e-5;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(17, 19, 23) && run_case(48, 40, 56);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) {
        printf("PASS %.3f\n", time_ms);
        return 0;
    }
    printf("FAIL matmul accuracy\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "stable_softmax_topk",
        "description": "Numerically stable softmax and top-k probability extraction",
        "category": "mixed_precision",
        "optimization_prompt": """Implement a numerically stable softmax top-k routine in C.

Function signature:
void stable_softmax_topk(const float* logits, int n, int k, int* out_indices, float* out_probs)

Requirements:
- Compute softmax probabilities over logits.
- Return the indices and probabilities of the top-k probabilities in descending probability order.
- Use max-subtraction for numerical stability.
- Use double for the exponential sum and normalization.
- Avoid O(n log n) full sorting if possible; O(n*k) selection is acceptable.
- Handle very large positive and negative logits without overflow, NaN, or Inf.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

void stable_softmax_topk(const float* logits, int n, int k, int* out_indices, float* out_probs);

static int ref_topk(const float* logits, int n, int k, int* idx, double* probs) {
    double maxv = -DBL_MAX;
    for (int i = 0; i < n; i++) if ((double)logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    double* all = (double*)malloc((size_t)n * sizeof(double));
    if (!all) return 0;
    for (int i = 0; i < n; i++) {
        all[i] = exp((double)logits[i] - maxv);
        sum += all[i];
    }
    for (int t = 0; t < k; t++) {
        int best = -1;
        double bestp = -1.0;
        for (int i = 0; i < n; i++) {
            int used = 0;
            for (int u = 0; u < t; u++) if (idx[u] == i) used = 1;
            double p = all[i] / sum;
            if (!used && (p > bestp || (fabs(p - bestp) < 1e-18 && i < best))) {
                best = i;
                bestp = p;
            }
        }
        idx[t] = best;
        probs[t] = bestp;
    }
    free(all);
    return 1;
}

int main(void) {
    const int n = 256;
    const int k = 5;
    float logits[n];
    for (int i = 0; i < n; i++) {
        logits[i] = (float)(((i * 37) % 101) - 50) / 3.0f;
    }
    logits[5] = 1000.0f;
    logits[77] = 999.0f;
    logits[123] = -1000.0f;
    logits[200] = 998.5f;

    int got_idx[k];
    float got_prob[k];
    int ref_idx[k];
    double ref_prob[k];

    clock_t start = clock();
    stable_softmax_topk(logits, n, k, got_idx, got_prob);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    if (!ref_topk(logits, n, k, ref_idx, ref_prob)) return 1;

    for (int t = 0; t < k; t++) {
        if (got_idx[t] != ref_idx[t]) {
            printf("FAIL index t=%d got=%d ref=%d\n", t, got_idx[t], ref_idx[t]);
            return 1;
        }
        if (!isfinite(got_prob[t]) || fabs((double)got_prob[t] - ref_prob[t]) > 1e-5) {
            printf("FAIL prob t=%d got=%.9g ref=%.9g\n", t, got_prob[t], ref_prob[t]);
            return 1;
        }
        if (t > 0 && got_prob[t] > got_prob[t - 1] + 1e-7f) {
            printf("FAIL ordering\n");
            return 1;
        }
    }

    printf("PASS %.3f\n", time_ms);
    return 0;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "rmsnorm_mixed",
        "description": "Mixed-precision RMSNorm kernel",
        "category": "mixed_precision",
        "optimization_prompt": """Implement an optimized mixed-precision RMSNorm kernel in C.

Function signature:
void rmsnorm_mixed(const float* x, const float* weight, float* y, int batch, int hidden, float eps)

For each row b:
  rms = sqrt(mean(x[b, j]^2) + eps)
  y[b, j] = x[b, j] / rms * weight[j]

Requirements:
- Use double accumulation for the sum of squares.
- Store output as float.
- Optimize memory access for row-major contiguous arrays.
- Correctly handle hidden sizes that are not multiples of 4 or 8.
- Do not change the mathematical definition.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void rmsnorm_mixed(const float* x, const float* weight, float* y, int batch, int hidden, float eps);

static int run_case(int batch, int hidden) {
    float* x = (float*)malloc((size_t)batch * hidden * sizeof(float));
    float* w = (float*)malloc((size_t)hidden * sizeof(float));
    float* y = (float*)malloc((size_t)batch * hidden * sizeof(float));
    double* r = (double*)malloc((size_t)batch * hidden * sizeof(double));
    if (!x || !w || !y || !r) return 0;

    for (int j = 0; j < hidden; j++) w[j] = 0.5f + (float)((j * 11) % 17) / 9.0f;
    for (int i = 0; i < batch * hidden; i++) {
        int v = (i * 37) % 211;
        x[i] = ((float)v - 105.0f) / 17.0f;
        if ((i % 97) == 0) x[i] *= 100.0f;
    }

    const float eps = 1e-5f;
    for (int b = 0; b < batch; b++) {
        double ss = 0.0;
        for (int j = 0; j < hidden; j++) {
            double xv = x[b * hidden + j];
            ss += xv * xv;
        }
        double inv = 1.0 / sqrt(ss / hidden + (double)eps);
        for (int j = 0; j < hidden; j++) r[b * hidden + j] = (double)x[b * hidden + j] * inv * (double)w[j];
    }

    rmsnorm_mixed(x, w, y, batch, hidden, eps);

    double max_abs = 0.0, err_norm = 0.0, ref_norm = 0.0;
    for (int i = 0; i < batch * hidden; i++) {
        double err = fabs((double)y[i] - r[i]);
        if (err > max_abs) max_abs = err;
        err_norm += err * err;
        ref_norm += r[i] * r[i];
    }
    free(x); free(w); free(y); free(r);
    return max_abs < 5e-5 && sqrt(err_norm / (ref_norm + 1e-30)) < 1e-6;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(7, 65) && run_case(16, 128);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) {
        printf("PASS %.3f\n", time_ms);
        return 0;
    }
    printf("FAIL rmsnorm accuracy\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "cg_step_mixed_csr",
        "description": "Mixed-precision Conjugate Gradient update on CSR matrix",
        "category": "sparse_linear_algebra",
        "optimization_prompt": """Implement one mixed-precision Conjugate Gradient update step for a symmetric positive definite sparse matrix in CSR format.

Function signature:
void cg_step_mixed_csr(int n, const int* row_ptr, const int* col_idx, const float* values,
                       const double* r, const double* p,
                       double* x, double* r_next, double* p_next)

Requirements:
- Matrix values are stored as float, but dot products and vector updates must use double.
- Compute Ap = A * p.
- alpha = dot(r, r) / dot(p, Ap)
- x = x + alpha * p
- r_next = r - alpha * Ap
- beta = dot(r_next, r_next) / dot(r, r)
- p_next = r_next + beta * p
- Avoid unnecessary repeated sparse matrix-vector products.
- Handle zero or tiny denominators defensively.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void cg_step_mixed_csr(int n, const int* row_ptr, const int* col_idx, const float* values,
                       const double* r, const double* p,
                       double* x, double* r_next, double* p_next);

static void reference(int n, const int* row_ptr, const int* col_idx, const float* values,
                      const double* r, const double* p, double* x, double* rn, double* pn) {
    double* Ap = (double*)calloc((size_t)n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) sum += (double)values[e] * p[col_idx[e]];
        Ap[i] = sum;
    }
    double rr = 0.0, pAp = 0.0;
    for (int i = 0; i < n; i++) {
        rr += r[i] * r[i];
        pAp += p[i] * Ap[i];
    }
    double alpha = rr / pAp;
    double rn2 = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] += alpha * p[i];
        rn[i] = r[i] - alpha * Ap[i];
        rn2 += rn[i] * rn[i];
    }
    double beta = rn2 / rr;
    for (int i = 0; i < n; i++) pn[i] = rn[i] + beta * p[i];
    free(Ap);
}

int main(void) {
    const int n = 64;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc((3 * n) * sizeof(int));
    float* values = (float*)malloc((3 * n) * sizeof(float));
    double *r = (double*)malloc(n * sizeof(double));
    double *p = (double*)malloc(n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *x_ref = (double*)malloc(n * sizeof(double));
    double *rn = (double*)malloc(n * sizeof(double));
    double *pn = (double*)malloc(n * sizeof(double));
    double *rn_ref = (double*)malloc(n * sizeof(double));
    double *pn_ref = (double*)malloc(n * sizeof(double));
    if (!row_ptr || !col_idx || !values || !r || !p || !x || !x_ref || !rn || !pn || !rn_ref || !pn_ref) return 1;

    int e = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) { col_idx[e] = i - 1; values[e++] = -1.0f; }
        col_idx[e] = i; values[e++] = 4.0f + (float)(i % 5) * 0.01f;
        if (i + 1 < n) { col_idx[e] = i + 1; values[e++] = -1.0f; }
        row_ptr[i + 1] = e;
    }

    for (int i = 0; i < n; i++) {
        r[i] = ((i * 17) % 31 - 15) / 7.0;
        p[i] = ((i * 13) % 29 - 14) / 9.0;
        x[i] = ((i * 11) % 23 - 11) / 19.0;
        x_ref[i] = x[i];
    }

    clock_t start = clock();
    cg_step_mixed_csr(n, row_ptr, col_idx, values, r, p, x, rn, pn);
    clock_t end = clock();
    reference(n, row_ptr, col_idx, values, r, p, x_ref, rn_ref, pn_ref);

    double err = 0.0, ref = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - x_ref[i];
        double dr = rn[i] - rn_ref[i];
        double dp = pn[i] - pn_ref[i];
        err += dx * dx + dr * dr + dp * dp;
        ref += x_ref[i] * x_ref[i] + rn_ref[i] * rn_ref[i] + pn_ref[i] * pn_ref[i];
    }
    double rel = sqrt(err / (ref + 1e-30));
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    free(row_ptr); free(col_idx); free(values); free(r); free(p); free(x); free(x_ref); free(rn); free(pn); free(rn_ref); free(pn_ref);
    if (rel < 1e-7) {
        printf("PASS %.3f\n", time_ms);
        return 0;
    }
    printf("FAIL cg rel=%.9g\n", rel);
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "stencil3d_mixed_7pt",
        "description": "Precision-sensitive 3D 7-point stencil",
        "category": "stencil",
        "optimization_prompt": """Implement an optimized 3D 7-point stencil in C.

Function signature:
void stencil3d_mixed_7pt(const float* in, float* out, int nx, int ny, int nz, const float coeff[7])

For each interior cell (x,y,z), compute:
out[z,y,x] = c0*center + c1*left + c2*right + c3*down + c4*up + c5*back + c6*front

Requirements:
- Arrays are flattened in row-major order: index = (z * ny + y) * nx + x.
- Copy boundary cells from input to output unchanged.
- Use double or carefully ordered accumulation to reduce numerical error.
- Optimize loop order for contiguous memory access.
- Handle small dimensions and non-cubic grids correctly.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void stencil3d_mixed_7pt(const float* in, float* out, int nx, int ny, int nz, const float coeff[7]);

static int idx3(int x, int y, int z, int nx, int ny) {
    return (z * ny + y) * nx + x;
}

static int run_case(int nx, int ny, int nz) {
    int total = nx * ny * nz;
    float* in = (float*)malloc((size_t)total * sizeof(float));
    float* out = (float*)malloc((size_t)total * sizeof(float));
    double* ref = (double*)malloc((size_t)total * sizeof(double));
    if (!in || !out || !ref) return 0;

    float c[7] = {0.43f, 0.07f, -0.03f, 0.11f, -0.05f, 0.19f, -0.09f};
    for (int i = 0; i < total; i++) {
        in[i] = ((float)((i * 37) % 257) - 128.0f) / 31.0f;
        out[i] = 7777.0f;
        ref[i] = in[i];
    }
    for (int z = 1; z < nz - 1; z++) {
        for (int y = 1; y < ny - 1; y++) {
            for (int x = 1; x < nx - 1; x++) {
                int id = idx3(x, y, z, nx, ny);
                double acc = 0.0;
                acc += (double)c[0] * in[id];
                acc += (double)c[1] * in[idx3(x - 1, y, z, nx, ny)];
                acc += (double)c[2] * in[idx3(x + 1, y, z, nx, ny)];
                acc += (double)c[3] * in[idx3(x, y - 1, z, nx, ny)];
                acc += (double)c[4] * in[idx3(x, y + 1, z, nx, ny)];
                acc += (double)c[5] * in[idx3(x, y, z - 1, nx, ny)];
                acc += (double)c[6] * in[idx3(x, y, z + 1, nx, ny)];
                ref[id] = acc;
            }
        }
    }

    stencil3d_mixed_7pt(in, out, nx, ny, nz, c);

    double max_abs = 0.0;
    for (int i = 0; i < total; i++) {
        double err = fabs((double)out[i] - ref[i]);
        if (err > max_abs) max_abs = err;
    }
    free(in); free(out); free(ref);
    return max_abs < 1e-5;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(17, 13, 11) && run_case(32, 24, 16);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) {
        printf("PASS %.3f\n", time_ms);
        return 0;
    }
    printf("FAIL stencil accuracy\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
]

if __name__ == "__main__":
    for problem in PAPER_SOURCE_NUMERICAL_PROBLEMS:
        print(f"{problem['name']} ({problem['category']}): {problem['description']}")
    print(f"Total problems: {len(PAPER_SOURCE_NUMERICAL_PROBLEMS)}")
