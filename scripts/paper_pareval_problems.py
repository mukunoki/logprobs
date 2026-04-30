"""ParEval-derived benchmarks adapted to the C-function test harness.

The source tasks are taken from the ParEval serial C++ prompts and rewritten as
plain C function-generation tasks so they can be evaluated by the existing gcc
test harness.
"""

PAPER_PAREVAL_PROBLEMS = [
    {
        "name": "pareval_convex_hull_perimeter_f64",
        "description": "ParEval geometry convex hull perimeter adapted to C",
        "category": "pareval_geometry",
        "optimization_prompt": """Implement a convex hull perimeter routine in C.
This benchmark is adapted from ParEval's serial convex hull perimeter task.

Type and function signature:
typedef struct ParevalPoint { double x; double y; } ParevalPoint;
double pareval_convex_hull_perimeter_f64(const ParevalPoint* points, int n)

Return the perimeter of the smallest convex polygon containing all input points.
Handle duplicate points and collinear points correctly.
If n <= 1, return 0.0. If all points are collinear, return twice the distance between the two extreme points.
The result is compared with a small floating-point tolerance.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct ParevalPoint { double x; double y; } ParevalPoint;

double pareval_convex_hull_perimeter_f64(const ParevalPoint* points, int n);

static int cmp_point(const void* a, const void* b) {
    const ParevalPoint* p = (const ParevalPoint*)a;
    const ParevalPoint* q = (const ParevalPoint*)b;
    if (p->x < q->x) return -1;
    if (p->x > q->x) return 1;
    if (p->y < q->y) return -1;
    if (p->y > q->y) return 1;
    return 0;
}

static double cross(ParevalPoint a, ParevalPoint b, ParevalPoint c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

static double dist(ParevalPoint a, ParevalPoint b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

static double ref_perimeter(const ParevalPoint* points, int n) {
    if (n <= 1) return 0.0;
    ParevalPoint* pts = (ParevalPoint*)malloc((size_t)n * sizeof(ParevalPoint));
    ParevalPoint* hull = (ParevalPoint*)malloc((size_t)(2 * n) * sizeof(ParevalPoint));
    if (!pts || !hull) exit(2);
    for (int i = 0; i < n; ++i) pts[i] = points[i];
    qsort(pts, (size_t)n, sizeof(ParevalPoint), cmp_point);

    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (m == 0 || pts[i].x != pts[m - 1].x || pts[i].y != pts[m - 1].y) {
            pts[m++] = pts[i];
        }
    }
    if (m <= 1) {
        free(pts);
        free(hull);
        return 0.0;
    }

    int k = 0;
    for (int i = 0; i < m; ++i) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.0) k--;
        hull[k++] = pts[i];
    }
    for (int i = m - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0.0) k--;
        hull[k++] = pts[i];
    }

    double perim = 0.0;
    for (int i = 0; i < k - 1; ++i) perim += dist(hull[i], hull[i + 1]);
    free(pts);
    free(hull);
    return perim;
}

static int run_case(const ParevalPoint* points, int n) {
    double got = pareval_convex_hull_perimeter_f64(points, n);
    double expect = ref_perimeter(points, n);
    double err = fabs(got - expect);
    double tol = 1e-7 * fmax(1.0, fabs(expect));
    if (!(err <= tol)) {
        printf("FAIL hull n=%d got %.12f expected %.12f err %.3g\n", n, got, expect, err);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    const ParevalPoint a[] = {{0,3},{1,1},{2,2},{4,4},{0,0},{1,2},{3,1},{3,3}};
    const ParevalPoint b[] = {{0,0},{1,0},{2,0},{3,0},{1,0},{2,0}};
    const ParevalPoint c[] = {{0,0},{2,0},{2,2},{0,2},{1,1},{0,0},{2,2}};
    const ParevalPoint d[] = {{-1,-1},{-2,3},{0,0},{4,1},{2,-3},{-3,-2},{1,4},{3,3}};
    const ParevalPoint e[] = {{5,5}};
    ok &= run_case(a, (int)(sizeof(a) / sizeof(a[0])));
    ok &= run_case(b, (int)(sizeof(b) / sizeof(b[0])));
    ok &= run_case(c, (int)(sizeof(c) / sizeof(c[0])));
    ok &= run_case(d, (int)(sizeof(d) / sizeof(d[0])));
    ok &= run_case(e, (int)(sizeof(e) / sizeof(e[0])));
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
        "name": "pareval_largest_component_i32",
        "description": "ParEval graph largest connected component adapted to C",
        "category": "pareval_graph",
        "optimization_prompt": """Implement a graph connected-component routine in C.
This benchmark is adapted from ParEval's serial largest component task.

Function signature:
int pareval_largest_component_i32(const int* adjacency, int n)

adjacency is an n x n row-major adjacency matrix for an undirected graph.
Return the number of vertices in the largest connected component.
Treat any nonzero adjacency entry as an edge. Handle n <= 0 by returning 0.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int pareval_largest_component_i32(const int* adjacency, int n);

static int ref_largest(const int* A, int n) {
    if (n <= 0) return 0;
    int* seen = (int*)calloc((size_t)n, sizeof(int));
    int* queue = (int*)malloc((size_t)n * sizeof(int));
    if (!seen || !queue) exit(2);
    int best = 0;
    for (int s = 0; s < n; ++s) {
        if (seen[s]) continue;
        int head = 0, tail = 0, count = 0;
        seen[s] = 1;
        queue[tail++] = s;
        while (head < tail) {
            int v = queue[head++];
            count++;
            for (int u = 0; u < n; ++u) {
                if (!seen[u] && A[(size_t)v * n + u] != 0) {
                    seen[u] = 1;
                    queue[tail++] = u;
                }
            }
        }
        if (count > best) best = count;
    }
    free(seen);
    free(queue);
    return best;
}

static int run_case(const int* A, int n) {
    int got = pareval_largest_component_i32(A, n);
    int expect = ref_largest(A, n);
    if (got != expect) {
        printf("FAIL largest component n=%d got %d expected %d\n", n, got, expect);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    const int a[] = {
        0,1,0,0,
        1,0,0,0,
        0,0,0,1,
        0,0,1,0
    };
    const int b[] = {
        0,1,1,0,0,0,
        1,0,0,0,0,0,
        1,0,0,0,0,0,
        0,0,0,0,1,0,
        0,0,0,1,0,1,
        0,0,0,0,1,0
    };
    const int c[] = {
        0,0,0,0,0,
        0,0,1,0,0,
        0,1,0,1,0,
        0,0,1,0,0,
        0,0,0,0,0
    };
    ok &= run_case(a, 4);
    ok &= run_case(b, 6);
    ok &= run_case(c, 5);
    ok &= (pareval_largest_component_i32(NULL, 0) == 0);
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
        "name": "pareval_sort_ignore_zero_i32",
        "description": "ParEval sort non-zero elements while leaving zero positions fixed",
        "category": "pareval_sort",
        "optimization_prompt": """Implement an integer array sorting routine in C.
This benchmark is adapted from ParEval's serial sort-non-zero-elements task.

Function signature:
void pareval_sort_ignore_zero_i32(int* x, int n)

Sort the nonzero elements of x in ascending order while leaving zero-valued positions fixed.
For example, [8, 4, 0, 9, 8, 0, 1, -1, 7] becomes [-1, 1, 0, 4, 7, 0, 8, 8, 9].
Handle n <= 0 and arrays with all zeros or no zeros.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void pareval_sort_ignore_zero_i32(int* x, int n);

static int cmp_int(const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

static void ref_sort(int* x, int n) {
    int* vals = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!vals) exit(2);
    int m = 0;
    for (int i = 0; i < n; ++i) if (x[i] != 0) vals[m++] = x[i];
    qsort(vals, (size_t)m, sizeof(int), cmp_int);
    int j = 0;
    for (int i = 0; i < n; ++i) if (x[i] != 0) x[i] = vals[j++];
    free(vals);
}

static int run_case(const int* input, int n) {
    int* got = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    int* expect = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!got || !expect) exit(2);
    memcpy(got, input, (size_t)n * sizeof(int));
    memcpy(expect, input, (size_t)n * sizeof(int));
    pareval_sort_ignore_zero_i32(got, n);
    ref_sort(expect, n);
    int ok = 1;
    for (int i = 0; i < n; ++i) {
        if (got[i] != expect[i]) {
            printf("FAIL sort n=%d idx=%d got %d expected %d\n", n, i, got[i], expect[i]);
            ok = 0;
            break;
        }
    }
    free(got);
    free(expect);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    const int a[] = {8, 4, 0, 9, 8, 0, 1, -1, 7};
    const int b[] = {0, 0, 0, 0};
    const int c[] = {5, -3, 2, -3, 9, 1};
    const int d[] = {0, -1, 0, -5, 0, 2, 2, 0, -9};
    ok &= run_case(a, (int)(sizeof(a) / sizeof(a[0])));
    ok &= run_case(b, (int)(sizeof(b) / sizeof(b[0])));
    ok &= run_case(c, (int)(sizeof(c) / sizeof(c[0])));
    ok &= run_case(d, (int)(sizeof(d) / sizeof(d[0])));
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
        "name": "pareval_game_of_life_i32",
        "description": "ParEval Game of Life one-step stencil adapted to C",
        "category": "pareval_stencil",
        "optimization_prompt": """Implement one generation of Conway's Game of Life in C.
This benchmark is adapted from ParEval's serial Game of Life task.

Function signature:
void pareval_game_of_life_i32(const int* input, int* output, int n)

input and output are n x n row-major grids. A live cell is 1 and a dead cell is 0.
Cells outside the grid are dead.
Rules:
- A live cell with fewer than 2 live neighbors dies.
- A live cell with 2 or 3 live neighbors remains alive.
- A live cell with more than 3 live neighbors dies.
- A dead cell with exactly 3 live neighbors becomes alive.
Handle n <= 0 without writing output.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void pareval_game_of_life_i32(const int* input, int* output, int n);

static void ref_life(const int* input, int* output, int n) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            int live = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int rr = r + dr;
                    int cc = c + dc;
                    if (rr >= 0 && rr < n && cc >= 0 && cc < n) {
                        live += input[(size_t)rr * n + cc] != 0;
                    }
                }
            }
            int cur = input[(size_t)r * n + c] != 0;
            output[(size_t)r * n + c] = (cur ? (live == 2 || live == 3) : (live == 3));
        }
    }
}

static int run_case(const int* input, int n) {
    int total = n * n;
    int* got = (int*)calloc((size_t)total, sizeof(int));
    int* expect = (int*)calloc((size_t)total, sizeof(int));
    if (!got || !expect) exit(2);
    pareval_game_of_life_i32(input, got, n);
    ref_life(input, expect, n);
    int ok = 1;
    for (int i = 0; i < total; ++i) {
        if ((got[i] != 0) != (expect[i] != 0)) {
            printf("FAIL life n=%d idx=%d got %d expected %d\n", n, i, got[i], expect[i]);
            ok = 0;
            break;
        }
    }
    free(got);
    free(expect);
    return ok;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    const int a[] = {
        0,0,0,0,0,
        0,1,0,0,0,
        0,1,1,0,0,
        0,0,1,1,0,
        0,1,0,0,0
    };
    const int b[] = {
        0,0,0,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,0,0,0,0
    };
    const int c[] = {
        1,1,0,
        1,1,0,
        0,0,0
    };
    const int d[] = {
        1,0,1,0,
        0,1,1,0,
        1,1,0,0,
        0,0,0,1
    };
    ok &= run_case(a, 5);
    ok &= run_case(b, 5);
    ok &= run_case(c, 3);
    ok &= run_case(d, 4);
    pareval_game_of_life_i32(NULL, NULL, 0);
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
