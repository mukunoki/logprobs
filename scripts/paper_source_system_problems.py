"""Paper system source problems for CBBA.

These tasks are intentionally more algorithmic than the simpler source benchmark sets.
They require non-trivial C kernels, careful boundary handling, and cache-friendly
or fused implementations.  The harnesses check correctness and print runtime;
the current CBBA scripts use pass/fail plus the reported runtime.
"""

PAPER_SOURCE_SYSTEM_PROBLEMS = [
    {
        "name": "conv2d_3x3_multi_channel",
        "description": "Multi-channel 3x3 valid convolution with cache-friendly loop order",
        "category": "stencil_convolution",
        "optimization_prompt": """Implement an optimized multi-channel 3x3 valid convolution in C.

Function signature:
void conv2d_3x3_multi_channel(const float* input, const float* kernel, float* output,
                              int height, int width, int channels)

Data layout:
- input is flattened H x W x C in row-major NHWC order: ((y * width + x) * channels + c)
- kernel is flattened 3 x 3 x C: ((ky * 3 + kx) * channels + c)
- output is flattened (H-2) x (W-2) in row-major order.

For every output pixel (y, x), compute the sum over ky=0..2, kx=0..2, c=0..channels-1:
  output[y, x] = input[y+ky, x+kx, c] * kernel[ky, kx, c]

Requirements:
- Use a cache-friendly loop order and avoid redundant index recomputation where practical.
- Accumulate in double or a carefully ordered float accumulation to reduce numerical error.
- Correctly handle dimensions that are not multiples of a vector width.
- Do not allocate a large temporary image.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void conv2d_3x3_multi_channel(const float* input, const float* kernel, float* output,
                              int height, int width, int channels);

static int idx(int y, int x, int c, int width, int channels) {
    return (y * width + x) * channels + c;
}

static int run_case(int h, int w, int c) {
    int out_h = h - 2, out_w = w - 2;
    size_t in_n = (size_t)h * w * c;
    size_t k_n = (size_t)3 * 3 * c;
    size_t out_n = (size_t)out_h * out_w;
    float* input = (float*)malloc(in_n * sizeof(float));
    float* kernel = (float*)malloc(k_n * sizeof(float));
    float* output = (float*)malloc(out_n * sizeof(float));
    double* ref = (double*)malloc(out_n * sizeof(double));
    if (!input || !kernel || !output || !ref) return 0;

    for (size_t i = 0; i < in_n; i++) input[i] = ((int)((i * 37) % 257) - 128) / 29.0f;
    for (size_t i = 0; i < k_n; i++) kernel[i] = ((int)((i * 17) % 31) - 15) / 23.0f;
    for (size_t i = 0; i < out_n; i++) { output[i] = 9999.0f; ref[i] = 0.0; }

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            double sum = 0.0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    for (int ch = 0; ch < c; ch++) {
                        sum += (double)input[idx(y + ky, x + kx, ch, w, c)] *
                               (double)kernel[(ky * 3 + kx) * c + ch];
                    }
                }
            }
            ref[y * out_w + x] = sum;
        }
    }

    conv2d_3x3_multi_channel(input, kernel, output, h, w, c);

    double max_abs = 0.0, err_norm = 0.0, ref_norm = 0.0;
    for (size_t i = 0; i < out_n; i++) {
        double err = fabs((double)output[i] - ref[i]);
        if (err > max_abs) max_abs = err;
        err_norm += err * err;
        ref_norm += ref[i] * ref[i];
    }
    free(input); free(kernel); free(output); free(ref);
    return max_abs < 2e-4 && sqrt(err_norm / (ref_norm + 1e-30)) < 2e-6;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(19, 23, 5) && run_case(48, 40, 13);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) { printf("PASS %.3f\n", time_ms); return 0; }
    printf("FAIL conv2d\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "csr_spmv_axpy_dot",
        "description": "Fused CSR sparse matrix-vector multiply, AXPY, and dot product",
        "category": "sparse_linear_algebra",
        "optimization_prompt": """Implement a fused CSR sparse matrix-vector kernel in C.

Function signature:
double csr_spmv_axpy_dot(int n, const int* row_ptr, const int* col_idx, const double* values,
                         const double* x, double alpha, double* y)

For a sparse matrix A in CSR format, compute:
  y[i] = alpha * (A*x)[i] + y[i]
and return dot(y, y) after the update.

Requirements:
- Traverse CSR rows only once.
- Use double accumulation.
- Avoid allocating a temporary vector for A*x.
- Handle empty rows correctly.
- Keep memory accesses cache-friendly.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double csr_spmv_axpy_dot(int n, const int* row_ptr, const int* col_idx, const double* values,
                         const double* x, double alpha, double* y);

static int run_case(int n) {
    int max_nnz = 5 * n;
    int* row_ptr = (int*)malloc((n + 1) * sizeof(int));
    int* col_idx = (int*)malloc(max_nnz * sizeof(int));
    double* values = (double*)malloc(max_nnz * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* y_ref = (double*)malloc(n * sizeof(double));
    if (!row_ptr || !col_idx || !values || !x || !y || !y_ref) return 0;

    int e = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        if (i % 11 == 0) { row_ptr[i + 1] = e; continue; }
        int cols[5] = {i, (i + 1) % n, (i * 7 + 3) % n, (i + n - 1) % n, (i * 13 + 5) % n};
        for (int t = 0; t < 5; t++) {
            col_idx[e] = cols[t];
            values[e] = ((i * 17 + t * 19) % 37 - 18) / 11.0;
            e++;
        }
        row_ptr[i + 1] = e;
    }
    for (int i = 0; i < n; i++) {
        x[i] = ((i * 23) % 41 - 20) / 13.0;
        y[i] = ((i * 29) % 43 - 21) / 17.0;
        y_ref[i] = y[i];
    }

    double alpha = -0.375;
    double ref_dot = 0.0;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) sum += values[p] * x[col_idx[p]];
        y_ref[i] += alpha * sum;
        ref_dot += y_ref[i] * y_ref[i];
    }

    double got_dot = csr_spmv_axpy_dot(n, row_ptr, col_idx, values, x, alpha, y);

    double max_abs = fabs(got_dot - ref_dot);
    for (int i = 0; i < n; i++) {
        double err = fabs(y[i] - y_ref[i]);
        if (err > max_abs) max_abs = err;
    }
    free(row_ptr); free(col_idx); free(values); free(x); free(y); free(y_ref);
    return max_abs < 1e-9;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(17) && run_case(1024);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) { printf("PASS %.3f\n", time_ms); return 0; }
    printf("FAIL csr_spmv_axpy_dot\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "radix_sort_u32_pairs",
        "description": "Stable radix sort of uint32 key/value pairs",
        "category": "sorting",
        "optimization_prompt": """Implement an optimized stable radix sort for uint32 key/value pairs in C.

Function signature:
void radix_sort_u32_pairs(unsigned int* keys, unsigned int* values, int n)

Requirements:
- Sort keys in ascending unsigned order.
- Move values along with their corresponding keys.
- The sort must be stable for equal keys.
- Use an O(n) radix/counting-sort style algorithm rather than comparison sort.
- Correctly handle n <= 1 and repeated keys.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void radix_sort_u32_pairs(unsigned int* keys, unsigned int* values, int n);

static unsigned int make_key(int i) {
    unsigned int x = (unsigned int)i * 2654435761u + 1013904223u;
    x ^= x >> 16;
    return x % 997u;
}

static int run_case(int n) {
    unsigned int* keys = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    unsigned int* values = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    unsigned int* orig_keys = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    unsigned int* orig_values = (unsigned int*)malloc((size_t)n * sizeof(unsigned int));
    if (!keys || !values || !orig_keys || !orig_values) return 0;
    for (int i = 0; i < n; i++) {
        keys[i] = make_key(n - 1 - i);
        values[i] = (unsigned int)i;
        orig_keys[i] = keys[i];
        orig_values[i] = values[i];
    }

    radix_sort_u32_pairs(keys, values, n);

    for (int i = 1; i < n; i++) {
        if (keys[i - 1] > keys[i]) { free(keys); free(values); free(orig_keys); free(orig_values); return 0; }
    }
    for (int i = 0; i < n; i++) {
        unsigned int v = values[i];
        if (v >= (unsigned int)n || orig_keys[v] != keys[i] || orig_values[v] != v) {
            free(keys); free(values); free(orig_keys); free(orig_values); return 0;
        }
        if (i > 0 && keys[i - 1] == keys[i] && values[i - 1] > values[i]) {
            free(keys); free(values); free(orig_keys); free(orig_values); return 0;
        }
    }
    free(keys); free(values); free(orig_keys); free(orig_values);
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(1) && run_case(257) && run_case(4096);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) { printf("PASS %.3f\n", time_ms); return 0; }
    printf("FAIL radix_sort_u32_pairs\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "floyd_warshall_blocked",
        "description": "Cache-aware all-pairs shortest paths on dense matrix",
        "category": "graph_algorithm",
        "optimization_prompt": """Implement an optimized Floyd-Warshall all-pairs shortest path kernel in C.

Function signature:
void floyd_warshall_blocked(double* dist, int n)

Data layout:
- dist is an n x n row-major matrix.
- dist[i*n+j] contains the current distance, or a large finite INF value.

Requirements:
- Update dist in-place to contain all-pairs shortest path distances.
- A blocked/cache-aware implementation is preferred, but correctness is most important.
- Do not overflow or produce NaN when paths contain INF-like large values.
- Correctly handle n that is not a multiple of the block size.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void floyd_warshall_blocked(double* dist, int n);

static void reference(double* d, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            double dik = d[i * n + k];
            for (int j = 0; j < n; j++) {
                double alt = dik + d[k * n + j];
                if (alt < d[i * n + j]) d[i * n + j] = alt;
            }
        }
    }
}

static int run_case(int n) {
    const double INF = 1e100;
    double* d = (double*)malloc((size_t)n * n * sizeof(double));
    double* r = (double*)malloc((size_t)n * n * sizeof(double));
    if (!d || !r) return 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double v = (i == j) ? 0.0 : INF;
            if (i != j && ((i * 17 + j * 13) % 5 != 0)) v = (double)((i * 11 + j * 7) % 23 + 1);
            d[i * n + j] = v;
            r[i * n + j] = v;
        }
    }
    floyd_warshall_blocked(d, n);
    reference(r, n);
    double max_abs = 0.0;
    for (int i = 0; i < n * n; i++) {
        double err = fabs(d[i] - r[i]);
        if (err > max_abs) max_abs = err;
    }
    free(d); free(r);
    return max_abs < 1e-9;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(7) && run_case(32);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) { printf("PASS %.3f\n", time_ms); return 0; }
    printf("FAIL floyd_warshall_blocked\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
    {
        "name": "nbody_tiled_step",
        "description": "One tiled N-body simulation step with softened inverse-square force",
        "category": "physics_kernel",
        "optimization_prompt": """Implement one optimized N-body simulation step in C.

Function signature:
void nbody_tiled_step(const double* pos, double* vel, double* out_pos,
                      int n, double dt, double softening)

Data layout:
- pos, vel, and out_pos are flattened n x 3 arrays.
- pos[3*i+0..2] is particle i position.
- vel[3*i+0..2] is particle i velocity; update vel in-place.

For each particle i, compute acceleration from all j != i:
  r = pos[j] - pos[i]
  inv = 1 / sqrt(dot(r,r) + softening)
  inv3 = inv * inv * inv
  acc += r * inv3
Then update:
  vel[i] += dt * acc
  out_pos[i] = pos[i] + dt * vel[i]

Requirements:
- Use double precision.
- Do not include self-interaction.
- Avoid modifying pos.
- A tiled/cache-friendly j-loop is preferred.
- Correctly handle small n.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void nbody_tiled_step(const double* pos, double* vel, double* out_pos,
                      int n, double dt, double softening);

static void reference(const double* pos, double* vel, double* out, int n, double dt, double eps) {
    double* new_vel = (double*)malloc((size_t)n * 3 * sizeof(double));
    for (int i = 0; i < n * 3; i++) new_vel[i] = vel[i];
    for (int i = 0; i < n; i++) {
        double ax = 0.0, ay = 0.0, az = 0.0;
        double pix = pos[3*i], piy = pos[3*i+1], piz = pos[3*i+2];
        for (int j = 0; j < n; j++) if (j != i) {
            double dx = pos[3*j] - pix;
            double dy = pos[3*j+1] - piy;
            double dz = pos[3*j+2] - piz;
            double inv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz + eps);
            double inv3 = inv * inv * inv;
            ax += dx * inv3;
            ay += dy * inv3;
            az += dz * inv3;
        }
        new_vel[3*i] += dt * ax;
        new_vel[3*i+1] += dt * ay;
        new_vel[3*i+2] += dt * az;
    }
    for (int i = 0; i < n; i++) {
        vel[3*i] = new_vel[3*i];
        vel[3*i+1] = new_vel[3*i+1];
        vel[3*i+2] = new_vel[3*i+2];
        out[3*i] = pos[3*i] + dt * vel[3*i];
        out[3*i+1] = pos[3*i+1] + dt * vel[3*i+1];
        out[3*i+2] = pos[3*i+2] + dt * vel[3*i+2];
    }
    free(new_vel);
}

static int run_case(int n) {
    double* pos = (double*)malloc((size_t)n * 3 * sizeof(double));
    double* vel = (double*)malloc((size_t)n * 3 * sizeof(double));
    double* vel_ref = (double*)malloc((size_t)n * 3 * sizeof(double));
    double* out = (double*)malloc((size_t)n * 3 * sizeof(double));
    double* out_ref = (double*)malloc((size_t)n * 3 * sizeof(double));
    if (!pos || !vel || !vel_ref || !out || !out_ref) return 0;
    for (int i = 0; i < n * 3; i++) {
        pos[i] = ((i * 37) % 211 - 105) / 31.0;
        vel[i] = ((i * 19) % 173 - 86) / 97.0;
        vel_ref[i] = vel[i];
        out[i] = out_ref[i] = 0.0;
    }
    double dt = 0.00125;
    double eps = 1e-2;
    nbody_tiled_step(pos, vel, out, n, dt, eps);
    reference(pos, vel_ref, out_ref, n, dt, eps);
    double max_abs = 0.0;
    for (int i = 0; i < n * 3; i++) {
        double e1 = fabs(vel[i] - vel_ref[i]);
        double e2 = fabs(out[i] - out_ref[i]);
        if (e1 > max_abs) max_abs = e1;
        if (e2 > max_abs) max_abs = e2;
    }
    free(pos); free(vel); free(vel_ref); free(out); free(out_ref);
    return max_abs < 1e-10;
}

int main(void) {
    clock_t start = clock();
    int ok = run_case(3) && run_case(96);
    clock_t end = clock();
    double time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    if (ok) { printf("PASS %.3f\n", time_ms); return 0; }
    printf("FAIL nbody_tiled_step\n");
    return 1;
}
''',
        "performance_threshold": None,
    },
]

if __name__ == "__main__":
    for problem in PAPER_SOURCE_SYSTEM_PROBLEMS:
        print(f"{problem['name']} ({problem['category']}): {problem['description']}")
    print(f"Total problems: {len(PAPER_SOURCE_SYSTEM_PROBLEMS)}")
