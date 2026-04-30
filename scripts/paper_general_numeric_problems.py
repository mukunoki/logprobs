"""General numerical benchmarks without artificial stride-heavy indexing."""

PAPER_GENERAL_NUMERIC_PROBLEMS = [
    {
        "name": "tridiagonal_solve_f64",
        "description": "Thomas algorithm for a tridiagonal linear system",
        "category": "linear_solver",
        "optimization_prompt": """Implement a tridiagonal linear solver in C.
Function signature:
int tridiagonal_solve_f64(const double* lower, const double* diag, const double* upper, const double* rhs, double* x, int n)

The matrix has lower diagonal lower[0..n-2], main diagonal diag[0..n-1], and upper diagonal upper[0..n-2].
Solve A*x = rhs using double precision and write the solution to x.
Return 1 on success and 0 if n <= 0 or a zero/tiny pivot makes the system numerically singular.
Do not modify lower, diag, upper, or rhs.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int tridiagonal_solve_f64(const double* lower, const double* diag, const double* upper, const double* rhs, double* x, int n);

static int run_case(int n) {
    double* lo = (double*)malloc((size_t)(n > 1 ? n - 1 : 1) * sizeof(double));
    double* di = (double*)malloc((size_t)n * sizeof(double));
    double* up = (double*)malloc((size_t)(n > 1 ? n - 1 : 1) * sizeof(double));
    double* rhs = (double*)malloc((size_t)n * sizeof(double));
    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* ref = (double*)malloc((size_t)n * sizeof(double));
    if (!lo || !di || !up || !rhs || !x || !ref) return 0;
    for (int i = 0; i < n; ++i) {
        ref[i] = sin(0.17 * (double)(i + 1)) + 0.03 * (double)(i % 5);
        di[i] = 4.0 + 0.01 * (double)(i % 7);
        x[i] = 999.0;
    }
    for (int i = 0; i + 1 < n; ++i) {
        lo[i] = -0.75 + 0.01 * (double)(i % 3);
        up[i] = -0.50 - 0.02 * (double)(i % 4);
    }
    for (int i = 0; i < n; ++i) {
        double v = di[i] * ref[i];
        if (i > 0) v += lo[i - 1] * ref[i - 1];
        if (i + 1 < n) v += up[i] * ref[i + 1];
        rhs[i] = v;
    }
    int ok = tridiagonal_solve_f64(lo, di, up, rhs, x, n);
    double err = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - ref[i];
        err += d * d;
        norm += ref[i] * ref[i];
    }
    free(lo); free(di); free(up); free(rhs); free(x); free(ref);
    return ok == 1 && sqrt(err / (norm + 1e-30)) < 1e-11;
}

int main(void) {
    clock_t start = clock();
    double lo[] = {1.0};
    double di_bad[] = {0.0};
    double up[] = {1.0};
    double rhs[] = {1.0};
    double x[] = {0.0};
    int ok = run_case(1) && run_case(8) && run_case(64);
    ok &= (tridiagonal_solve_f64(lo, di_bad, up, rhs, x, 1) == 0);
    ok &= (tridiagonal_solve_f64(lo, di_bad, up, rhs, x, 0) == 0);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL tridiagonal\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "natural_cubic_spline_f64",
        "description": "natural cubic spline second-derivative setup",
        "category": "interpolation",
        "optimization_prompt": """Implement natural cubic spline preprocessing in C.
Function signature:
int natural_cubic_spline_f64(const double* x, const double* y, double* y2, int n)

x[0..n-1] are sample positions and must be strictly increasing.
Compute the natural cubic spline second derivatives y2[0..n-1], with y2[0] = y2[n-1] = 0.
Return 1 on success and 0 if n < 2, x is not strictly increasing, or a numerical denominator is zero/tiny.
Use double precision and do not modify x or y.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int natural_cubic_spline_f64(const double* x, const double* y, double* y2, int n);

static int ref_spline(const double* x, const double* y, double* y2, int n) {
    double* u = (double*)calloc((size_t)n, sizeof(double));
    if (!u || n < 2) return 0;
    y2[0] = 0.0;
    u[0] = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        double h0 = x[i] - x[i - 1];
        double h1 = x[i + 1] - x[i];
        if (!(h0 > 0.0) || !(h1 > 0.0)) { free(u); return 0; }
        double sig = h0 / (h0 + h1);
        double p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        double dd = (y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0;
        u[i] = (6.0 * dd / (h0 + h1) - sig * u[i - 1]) / p;
    }
    y2[n - 1] = 0.0;
    for (int k = n - 2; k >= 0; --k) y2[k] = y2[k] * y2[k + 1] + u[k];
    free(u);
    return 1;
}

static int run_case(int n) {
    double* x = (double*)malloc((size_t)n * sizeof(double));
    double* y = (double*)malloc((size_t)n * sizeof(double));
    double* got = (double*)malloc((size_t)n * sizeof(double));
    double* ref = (double*)malloc((size_t)n * sizeof(double));
    if (!x || !y || !got || !ref) return 0;
    double pos = 0.0;
    for (int i = 0; i < n; ++i) {
        pos += 0.5 + 0.07 * (double)((i * 5) % 4);
        x[i] = pos;
        y[i] = sin(0.4 * x[i]) + 0.1 * x[i] * x[i];
        got[i] = 123.0;
    }
    int ok1 = natural_cubic_spline_f64(x, y, got, n);
    int ok2 = ref_spline(x, y, ref, n);
    double err = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        err += d * d;
        norm += ref[i] * ref[i];
    }
    free(x); free(y); free(got); free(ref);
    return ok1 == 1 && ok2 == 1 && sqrt(err / (norm + 1e-30)) < 1e-11;
}

int main(void) {
    clock_t start = clock();
    double xb[] = {0.0, 1.0, 1.0};
    double yb[] = {0.0, 1.0, 2.0};
    double y2b[] = {0.0, 0.0, 0.0};
    int ok = run_case(2) && run_case(7) && run_case(31);
    ok &= (natural_cubic_spline_f64(xb, yb, y2b, 3) == 0);
    ok &= (natural_cubic_spline_f64(xb, yb, y2b, 1) == 0);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    printf("FAIL spline\n");
    return 1;
}
""",
        "performance_threshold": 1.0,
    },
    {
        "name": "quadratic_roots_stable_f64",
        "description": "numerically stable real roots of a quadratic or linear equation",
        "category": "root_finding",
        "optimization_prompt": """Implement a numerically stable quadratic root solver in C.
Function signature:
int quadratic_roots_stable_f64(double a, double b, double c, double* r0, double* r1)

Solve a*x*x + b*x + c = 0 for real roots.
Return the number of real roots: 0, 1, or 2.
If a is zero or tiny, treat the equation as linear b*x + c = 0.
Write roots in ascending order to r0 and r1 when roots exist.
Use a stable formula that avoids catastrophic cancellation for large |b|.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <math.h>
#include <time.h>

int quadratic_roots_stable_f64(double a, double b, double c, double* r0, double* r1);

static int check(double a, double b, double c, int expect_n, double e0, double e1, const char* name) {
    double r0 = 99.0, r1 = -99.0;
    int n = quadratic_roots_stable_f64(a, b, c, &r0, &r1);
    if (n != expect_n) {
        printf("FAIL roots %s count got %d expected %d\n", name, n, expect_n);
        return 0;
    }
    if (n >= 1 && fabs(r0 - e0) > 1e-9 * fmax(1.0, fabs(e0))) {
        printf("FAIL roots %s r0 got %.17g expected %.17g\n", name, r0, e0);
        return 0;
    }
    if (n >= 2 && fabs(r1 - e1) > 1e-9 * fmax(1.0, fabs(e1))) {
        printf("FAIL roots %s r1 got %.17g expected %.17g\n", name, r1, e1);
        return 0;
    }
    double scale0 = fmax(1.0, fabs(a*r0*r0) + fabs(b*r0) + fabs(c));
    if (n >= 1 && fabs(a*r0*r0 + b*r0 + c) > 1e-12 * scale0) {
        printf("FAIL roots %s residual0\n", name);
        return 0;
    }
    double scale1 = fmax(1.0, fabs(a*r1*r1) + fabs(b*r1) + fabs(c));
    if (n >= 2 && fabs(a*r1*r1 + b*r1 + c) > 1e-12 * scale1) {
        printf("FAIL roots %s residual1\n", name);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    ok &= check(1.0, -5.0, 6.0, 2, 2.0, 3.0, "simple");
    ok &= check(1.0, 2.0, 1.0, 1, -1.0, -1.0, "double");
    ok &= check(1.0, 0.0, 1.0, 0, 0.0, 0.0, "none");
    ok &= check(0.0, 2.0, -8.0, 1, 4.0, 4.0, "linear");
    ok &= check(1.0, 1.0e8, 1.0, 2, -1.0e8, -1.0e-8, "cancellation");
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
        "name": "eig2x2_symmetric_f64",
        "description": "eigendecomposition of a 2x2 symmetric matrix",
        "category": "linear_algebra",
        "optimization_prompt": """Implement eigendecomposition for a 2x2 symmetric matrix in C.
Function signature:
void eig2x2_symmetric_f64(const double* A, double* evals, double* evecs)

A is row-major [a00, a01, a10, a11] and should be treated as symmetric using the off-diagonal average.
Write eigenvalues to evals[0], evals[1] in descending order.
Write the corresponding unit eigenvectors as row-major columns in evecs:
evecs[0*2 + k] is the first component of eigenvector k and evecs[1*2 + k] is the second component.
Handle diagonal, repeated, nearly diagonal, and large-magnitude matrices robustly.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <math.h>
#include <time.h>

void eig2x2_symmetric_f64(const double* A, double* evals, double* evecs);

static int run_case(double a, double b, double d, const char* name) {
    double A[4] = {a, b, b, d};
    double evals[2] = {0.0, 0.0};
    double V[4] = {0.0, 0.0, 0.0, 0.0};
    eig2x2_symmetric_f64(A, evals, V);
    if (evals[0] + 1e-10 < evals[1]) {
        printf("FAIL eig %s ordering\n", name);
        return 0;
    }
    for (int k = 0; k < 2; ++k) {
        double v0 = V[0 * 2 + k];
        double v1 = V[1 * 2 + k];
        double norm = hypot(v0, v1);
        if (fabs(norm - 1.0) > 1e-10) {
            printf("FAIL eig %s norm %d %.17g\n", name, k, norm);
            return 0;
        }
        double r0 = a * v0 + b * v1 - evals[k] * v0;
        double r1 = b * v0 + d * v1 - evals[k] * v1;
        double scale = fmax(1.0, fmax(fabs(a), fmax(fabs(b), fabs(d))));
        if (hypot(r0, r1) > 1e-9 * scale) {
            printf("FAIL eig %s residual %d %.17g\n", name, k, hypot(r0, r1));
            return 0;
        }
    }
    double dot = V[0] * V[1] + V[2] * V[3];
    if (fabs(dot) > 1e-9) {
        printf("FAIL eig %s orthogonal %.17g\n", name, dot);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    ok &= run_case(3.0, 1.0, 2.0, "basic");
    ok &= run_case(5.0, 0.0, -1.0, "diagonal");
    ok &= run_case(2.0, 0.0, 2.0, "repeated");
    ok &= run_case(1.0e12, 3.0e6, 1.0e12 + 4.0, "large");
    ok &= run_case(-4.0, 2.5, 7.0, "mixed");
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
