"""Probe benchmarks for exploring when logprob-based ranking is effective."""

PAPER_PROBE_PROBLEMS = [
    {
        "name": "mixed_precision_dot_i8_f32",
        "description": "Quantized int8 dot product with mixed-precision accumulation",
        "category": "probe_mixed_precision",
        "optimization_prompt": """Implement this C function.

Function signature:
float mixed_precision_dot_i8_f32(const signed char* a, const signed char* b, int n,
                                 float scale_a, float scale_b,
                                 int zero_a, int zero_b)

For each element, subtract the zero point:
  da = (int)a[i] - zero_a
  db = (int)b[i] - zero_b
Return scale_a * scale_b * sum_i (da * db).

Handle n <= 0 by returning 0.0f.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

float mixed_precision_dot_i8_f32(const signed char* a, const signed char* b, int n,
                                 float scale_a, float scale_b,
                                 int zero_a, int zero_b);

static float ref_dot(const signed char* a, const signed char* b, int n,
                     float scale_a, float scale_b, int zero_a, int zero_b) {
    if (n <= 0) return 0.0f;
    long double acc = 0.0L;
    for (int i = 0; i < n; ++i) {
        long double da = (long double)((int)a[i] - zero_a);
        long double db = (long double)((int)b[i] - zero_b);
        acc += da * db;
    }
    return (float)((long double)scale_a * (long double)scale_b * acc);
}

static int run_case(const signed char* a, const signed char* b, int n,
                    float scale_a, float scale_b, int zero_a, int zero_b) {
    float got = mixed_precision_dot_i8_f32(a, b, n, scale_a, scale_b, zero_a, zero_b);
    float expect = ref_dot(a, b, n, scale_a, scale_b, zero_a, zero_b);
    float err = fabsf(got - expect);
    float tol = 2.0e-5f * fmaxf(1.0f, fabsf(expect));
    if (!(err <= tol)) {
        printf("FAIL n=%d got %.9g expected %.9g err %.9g tol %.9g\n", n, got, expect, err, tol);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;

    signed char small_a[] = {127, -128, 7, 0, 31, -9, 64, -64};
    signed char small_b[] = {-128, 127, -5, 3, 11, -17, -64, 64};
    ok &= run_case(small_a, small_b, 8, 0.03125f, 0.015625f, -3, 5);

    const int n_large = 200000;
    signed char* a = (signed char*)malloc((size_t)n_large);
    signed char* b = (signed char*)malloc((size_t)n_large);
    if (!a || !b) return 2;

    for (int i = 0; i < n_large; ++i) {
        a[i] = (signed char)((i * 37 + 91) % 256 - 128);
        b[i] = (signed char)((i * 53 + 17) % 256 - 128);
    }
    ok &= run_case(a, b, n_large, 1.0e-4f, -2.0e-4f, -7, 11);

    for (int i = 0; i < n_large; ++i) {
        a[i] = 127;
        b[i] = 127;
    }
    ok &= run_case(a, b, n_large, 1.0e-4f, 2.0e-4f, -5, 3);

    for (int i = 0; i < n_large; ++i) {
        a[i] = (i & 1) ? -128 : 127;
        b[i] = (i % 3 == 0) ? 127 : -128;
    }
    ok &= run_case(a, b, n_large, -3.0e-5f, 4.0e-4f, 13, -9);

    ok &= (mixed_precision_dot_i8_f32(NULL, NULL, 0, 1.0f, 1.0f, 0, 0) == 0.0f);

    free(a);
    free(b);
    double ms = 1000.0 * (double)(clock() - start) / CLOCKS_PER_SEC;
    if (ok) {
        printf("PASS %.3f\n", ms);
        return 0;
    }
    return 1;
}
""",
        "performance_threshold": 3.0,
    },
    {
        "name": "percent_decode_strict",
        "description": "strict percent-decoder with hexadecimal validation",
        "category": "probe_parser",
        "optimization_prompt": """Implement this C function.

Function signature:
int percent_decode_strict(const char* src, int n, unsigned char* dst)

Decode percent escapes in src[0..n-1].
A sequence %HH decodes to one byte, where H is 0-9, A-F, or a-f.
All non-percent bytes are copied as-is.
Return the decoded length.
Return -1 if n < 0, src or dst is NULL, a percent escape is incomplete, or a hex digit is invalid.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <string.h>
#include <time.h>

int percent_decode_strict(const char* src, int n, unsigned char* dst);

static int run_case(const char* src, const unsigned char* expect, int expect_len) {
    unsigned char out[256];
    for (int i = 0; i < 256; ++i) out[i] = 0xA5u;
    int got = percent_decode_strict(src, (int)strlen(src), out);
    if (got != expect_len) {
        printf("FAIL decode count '%s' got %d expected %d\n", src, got, expect_len);
        return 0;
    }
    if (expect_len >= 0 && memcmp(out, expect, (size_t)expect_len) != 0) {
        printf("FAIL decode bytes '%s'\n", src);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    unsigned char out[8];
    const unsigned char e1[] = "abc def";
    const unsigned char e2[] = {0x00, 0x2f, 0x7f, 0xff};
    const unsigned char e3[] = "100% ok";
    int ok = 1;
    ok &= run_case("abc%20def", e1, 7);
    ok &= run_case("%00%2f%7F%ff", e2, 4);
    ok &= run_case("100%25%20ok", e3, 7);
    ok &= run_case("%", NULL, -1);
    ok &= run_case("%4", NULL, -1);
    ok &= run_case("%xz", NULL, -1);
    ok &= run_case("abc%2G", NULL, -1);
    ok &= (percent_decode_strict(NULL, 3, out) == -1);
    ok &= (percent_decode_strict("abc", -1, out) == -1);
    ok &= (percent_decode_strict("abc", 3, NULL) == -1);
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
        "name": "csv_parse_quoted_fields",
        "description": "strict CSV field counter with quoted-field validation",
        "category": "probe_parser",
        "optimization_prompt": """Implement this C function.

Function signature:
int csv_parse_quoted_fields(const char* s, int n)

Count fields in one CSV record s[0..n-1].
Comma separates fields only when it is outside quotes.
A quoted field starts with " at the beginning of a field and ends with a matching ".
Inside a quoted field, two consecutive quotes "" represent an escaped quote.
After a closing quote, the next character must be a comma or the end of the record.
Return the number of fields, or -1 if n < 0, s is NULL, or the record is malformed.
An empty record has one empty field.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <string.h>
#include <time.h>

int csv_parse_quoted_fields(const char* s, int n);

static int check(const char* s, int expect) {
    int got = csv_parse_quoted_fields(s, (int)strlen(s));
    if (got != expect) {
        printf("FAIL csv '%s' got %d expected %d\n", s, got, expect);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    int ok = 1;
    ok &= check("", 1);
    ok &= check("a,b,c", 3);
    ok &= check("a,,c,", 4);
    ok &= check("\"a,b\",c", 2);
    ok &= check("\"a\"\"b\",c", 2);
    ok &= check("\"\",plain,\"x,y,z\"", 3);
    ok &= check("\"unterminated", -1);
    ok &= check("a,\"bad\"tail,c", -1);
    ok &= check("a,\"bad\" \"space\"", -1);
    ok &= check("a,b\"c,d", -1);
    ok &= (csv_parse_quoted_fields(NULL, 0) == -1);
    ok &= (csv_parse_quoted_fields("abc", -1) == -1);
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
        "name": "roi_crop_clamp_u8",
        "description": "image ROI crop with out-of-bounds padding",
        "category": "probe_indexing",
        "optimization_prompt": """Implement this C function.

Function signature:
void roi_crop_clamp_u8(const unsigned char* src, int width, int height, int channels,
                       int x0, int y0, int crop_w, int crop_h,
                       unsigned char pad, unsigned char* dst)

src is a row-major height x width x channels image.
For each output pixel (x, y), read source coordinate (x0 + x, y0 + y).
If the source coordinate is inside the image, copy all channels from src to dst.
Otherwise write pad to all channels.
dst is row-major crop_h x crop_w x channels.
If any dimension is non-positive or src or dst is NULL, do nothing.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <string.h>
#include <time.h>

void roi_crop_clamp_u8(const unsigned char* src, int width, int height, int channels,
                       int x0, int y0, int crop_w, int crop_h,
                       unsigned char pad, unsigned char* dst);

static unsigned char pix(int x, int y, int c) {
    return (unsigned char)(10 * y + 3 * x + c + 1);
}

static int run_case(int x0, int y0, int cw, int ch, unsigned char pad) {
    const int w = 5, h = 4, c = 3;
    unsigned char src[w * h * c];
    unsigned char dst[8 * 8 * 3];
    unsigned char ref[8 * 8 * 3];
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int k = 0; k < c; ++k)
                src[(y * w + x) * c + k] = pix(x, y, k);
    memset(dst, 0xEE, sizeof(dst));
    memset(ref, 0xEE, sizeof(ref));
    for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
            int sx = x0 + x;
            int sy = y0 + y;
            for (int k = 0; k < c; ++k) {
                unsigned char v = pad;
                if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                    v = src[(sy * w + sx) * c + k];
                }
                ref[(y * cw + x) * c + k] = v;
            }
        }
    }
    roi_crop_clamp_u8(src, w, h, c, x0, y0, cw, ch, pad, dst);
    int n = cw * ch * c;
    if (memcmp(dst, ref, (size_t)n) != 0) {
        printf("FAIL roi x0=%d y0=%d cw=%d ch=%d\n", x0, y0, cw, ch);
        return 0;
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    unsigned char guard[4] = {1, 2, 3, 4};
    int ok = 1;
    ok &= run_case(1, 1, 3, 2, 99);
    ok &= run_case(-2, -1, 4, 4, 77);
    ok &= run_case(3, 2, 4, 3, 55);
    roi_crop_clamp_u8(NULL, 5, 4, 3, 0, 0, 2, 2, 9, guard);
    ok &= (guard[0] == 1 && guard[1] == 2 && guard[2] == 3 && guard[3] == 4);
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
        "name": "topk_ignore_nan_f32",
        "description": "top-k indices with NaN filtering and deterministic ties",
        "category": "probe_selection",
        "optimization_prompt": """Implement this C function.

Function signature:
int topk_ignore_nan_f32(const float* x, int n, int k, int* indices)

Select indices of the largest values in x.
Ignore NaN values.
Write at most k indices to indices[0..].
The output order is descending by value; ties are broken by smaller index first.
Return the number of indices written.
Return 0 if x or indices is NULL, n <= 0, or k <= 0.

Return only C code.
Required #include directives and helper definitions are allowed.
Do not include markdown fences or explanatory text.""",
        "test_code": r"""
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

int topk_ignore_nan_f32(const float* x, int n, int k, int* indices);

static int check(const float* x, int n, int k, const int* expect, int expect_count, const char* name) {
    int out[16];
    for (int i = 0; i < 16; ++i) out[i] = -99;
    int got = topk_ignore_nan_f32(x, n, k, out);
    if (got != expect_count) {
        printf("FAIL topk %s count got %d expected %d\n", name, got, expect_count);
        return 0;
    }
    for (int i = 0; i < got; ++i) {
        if (out[i] != expect[i]) {
            printf("FAIL topk %s pos %d got %d expected %d\n", name, i, out[i], expect[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    clock_t start = clock();
    const float nanv = NAN;
    const float a[] = {1.0f, nanv, 3.0f, 2.0f, 3.0f, -5.0f};
    const int ea[] = {2, 4, 3};
    const float b[] = {nanv, nanv, -1.0f};
    const int eb[] = {2};
    const float c[] = {5.0f, 5.0f, 4.0f, 6.0f, 6.0f};
    const int ec[] = {3, 4, 0, 1, 2};
    int ok = 1;
    ok &= check(a, 6, 3, ea, 3, "a");
    ok &= check(b, 3, 5, eb, 1, "b");
    ok &= check(c, 5, 5, ec, 5, "c");
    ok &= (topk_ignore_nan_f32(NULL, 4, 2, (int*)ec) == 0);
    ok &= (topk_ignore_nan_f32(c, 5, 0, (int*)ec) == 0);
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
