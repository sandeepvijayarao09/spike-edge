/* Host latency benchmark (this is NOT a microcontroller measurement).
 * usage: bench in.bin   (same input format as harness.c) */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "spike_edge.h"
#include MODEL_HEADER

static double now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static int cmp(const void *a, const void *b) { double d = *(double *)a - *(double *)b; return (d > 0) - (d < 0); }
static volatile int32_t sink;

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb"); int32_t h[4];
    if (!f || fread(h, 4, 4, f) != 4) return 1;
    int16_t *raw = malloc(2 * h[0]);
    if (fread(raw, 2, h[0], f) != (size_t)h[0]) return 1;
    fclose(f);
    enum { REP = 15 };
    double t[3][REP]; long n_cand = 0;
    for (int mode = 0; mode < 3; mode++) {
        for (int r = 0; r < REP; r++) {
            static se_state_t s; se_event_t ev; int32_t acc = 0; long nc = 0;
            se_init(&s, h[1], h[2], h[3], mode == 2 ? &SE_MODEL : NULL);
            double t0 = now_ns();
            if (mode == 0) for (int32_t i = 0; i < h[0]; i++) acc += se_frontend(&s, raw[i]);
            else for (int32_t i = 0; i < h[0]; i++) nc += se_step(&s, raw[i], 0, &ev);
            t[mode][r] = (now_ns() - t0) / h[0];
            sink = acc; n_cand = nc;
        }
        qsort(t[mode], REP, sizeof(double), cmp);
    }
    /* per-inference latency */
    int8_t x[SE_WIN]; double ti[REP];
    for (int r = 0; r < REP; r++) {
        double t0 = now_ns(); int32_t a = 0;
        for (int i = 0; i < 100000; i++) { for (int j = 0; j < SE_WIN; j++) x[j] = (int8_t)((i * 7 + j * 13) & 63) - 32; a += se_model_run(&SE_MODEL, x); }
        ti[r] = (now_ns() - t0) / 100000; sink = a;
    }
    qsort(ti, REP, sizeof(double), cmp);
    printf("{\"ns_per_sample_frontend\": %.3f, \"ns_per_sample_threshold\": %.3f, \"ns_per_sample_gated_model\": %.3f, "
           "\"ns_per_inference\": %.2f, \"candidates\": %ld, \"samples\": %d, \"sizeof_state\": %zu}\n",
           t[0][REP / 2], t[1][REP / 2], t[2][REP / 2], ti[REP / 2], n_cand, h[0], sizeof(se_state_t));
    return 0;
}
