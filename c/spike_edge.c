#include "spike_edge.h"

/* The integer pipeline relies on >> of a negative int32/int64 being an arithmetic
 * (floor) shift, as numpy and every mainstream compiler for ARM/x86 implement it.
 * C99 leaves it implementation-defined, so refuse to build where it is not. */
typedef char se_requires_arithmetic_shift[((-1 >> 1) == -1 && ((int32_t)-5 >> 1) == -3) ? 1 : -1];

/* Wrap-safe "a is at or after b" for free-running uint32 sample counters
 * (valid while the two are within 2^31 samples, i.e. ~29.8 h of each other). */
static inline int se_after_eq(uint32_t a, uint32_t b) { return (int32_t)(a - b) >= 0; }

#ifndef SE_MAX_ACT
#define SE_MAX_ACT 128        /* bytes per activation buffer (largest C*L); checked by export_c.py */
#endif

void se_init_at(se_state_t *s, int32_t gate_thr, int32_t gain, int32_t logit_thr, const se_model_t *m, uint32_t n0) {
    uint8_t *p = (uint8_t *)s;
    for (unsigned i = 0; i < sizeof(*s); i++) p[i] = 0;
    s->gate_thr = gate_thr; s->gain = gain; s->logit_thr = logit_thr; s->model = m;
    s->n = s->n0 = s->dead_until = n0;
}

void se_init(se_state_t *s, int32_t gate_thr, int32_t gain, int32_t logit_thr, const se_model_t *m) {
    se_init_at(s, gate_thr, gain, logit_thr, m, 0);
}

int32_t se_frontend(se_state_t *s, int16_t x) {
    int32_t xs = (int32_t)x * 256;              /* x << 8 without UB on negatives */
    if (!s->started) { s->s1 = xs; s->started = 1; }
    s->s1 += (xs - s->s1) >> SE_KS;
    int32_t h1 = xs - s->s1;
    s->s2 += (h1 - s->s2) >> SE_KS;
    int32_t h2 = h1 - s->s2;
    s->s3 += (h2 - s->s3) >> SE_KF;
    return s->s3 >> 6;
}

static inline int8_t clamp_relu(int32_t v) { return (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v)); }

static inline int32_t requant(int32_t acc, int32_t m, int32_t sh) {
    int64_t r = ((int64_t)acc * m + ((int64_t)1 << (sh - 1))) >> sh;
    return (int32_t)(r < -128 ? -128 : (r > 127 ? 127 : r));
}

int32_t se_model_run(const se_model_t *m, const int8_t *x) {
    int8_t buf[2][SE_MAX_ACT];
    const int8_t *in = x;
    int8_t *out = buf[0];
    for (uint8_t li = 0; li < m->n_layers; li++) {
        const se_layer_t *L = &m->layers[li];
        int last = (li == m->n_layers - 1);
        if (L->type == SE_CONV) {
            for (int co = 0; co < L->cout; co++) {
                const int8_t *wc = L->w + co * L->cin * L->k;
                for (int l = 0; l < L->lout; l++) {
                    int32_t acc = L->b[co];
                    for (int ci = 0; ci < L->cin; ci++) {
                        const int8_t *xi = in + ci * L->lin + l * L->stride;
                        const int8_t *wi = wc + ci * L->k;
                        for (int k = 0; k < L->k; k++) acc += (int32_t)wi[k] * xi[k];
                    }
                    out[co * L->lout + l] = clamp_relu(requant(acc, L->m[co], L->sh[co]));
                }
            }
        } else {
            for (int co = 0; co < L->cout; co++) {
                const int8_t *wc = L->w + co * L->lin;
                int32_t acc = L->b[co];
                for (int i = 0; i < L->lin; i++) acc += (int32_t)wc[i] * in[i];
                if (last) return acc;
                out[co] = clamp_relu(requant(acc, L->m[co], L->sh[co]));
            }
        }
        in = out;
        out = (out == buf[0]) ? buf[1] : buf[0];
    }
    return 0;
}

static void classify(se_state_t *s, se_event_t *ev) {
    ev->idx = s->pend_i; ev->logit = 0; ev->accepted = 1;
    if (!s->model) return;
    int8_t x[SE_WIN];
    uint32_t start = s->pend_i - SE_PRE;
    for (int j = 0; j < SE_WIN; j++) {
        int32_t v = ((int32_t)s->ring[(start + j) & (SE_RING - 1)] * s->gain + 128) >> 8;
        x[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
    ev->logit = se_model_run(s->model, x);
    ev->accepted = ev->logit > s->logit_thr;
}

int se_step(se_state_t *s, int16_t xraw, int16_t *y_out, se_event_t *ev) {
    int32_t y = se_frontend(s, xraw);
    uint32_t i = s->n++;
    s->ring[i & (SE_RING - 1)] = (int16_t)y;
    if (y_out) *y_out = (int16_t)y;
    if (s->in_ev) {
        if (y < s->best) { s->best = y; s->best_i = i; }
        if (se_after_eq(i, s->end)) {
            s->in_ev = 0;
            s->dead_until = s->best_i + SE_DEAD;
            if (s->best_i - s->n0 >= SE_PRE) { s->has_pending = 1; s->pend_i = s->best_i; }
        }
    } else if (se_after_eq(i, s->dead_until) && y < -s->gate_thr) {
        s->in_ev = 1; s->best = y; s->best_i = i; s->end = i + SE_SEARCH;
    }
    if (s->has_pending && i == s->pend_i + SE_POST - 1) {
        s->has_pending = 0;
        classify(s, ev);
        return 1;
    }
    return 0;
}
