/* spike-edge: integer front-end + gated int8 spike classifier, plain C99.
 * No floats, no malloc, no libc beyond <stdint.h>. Mirrors the Python package (spike_edge) bit for bit. */
#ifndef SPIKE_EDGE_H
#define SPIKE_EDGE_H
#include <stdint.h>

#define SE_KS 4
#define SE_KF 1
#define SE_SEARCH 15
#define SE_DEAD 20
#define SE_PRE 10
#define SE_POST 22
#define SE_WIN (SE_PRE + SE_POST)
#define SE_RING 32            /* power of two >= SE_WIN: the classifier needs exactly the last 32 samples */

enum { SE_CONV = 0, SE_DENSE = 1 };

typedef struct {
    uint8_t type, cin, cout, k, stride;
    uint16_t lin, lout;       /* for dense: lin = flattened inputs, lout = 1 */
    const int8_t *w;
    const int32_t *b;
    const int32_t *m, *sh;    /* requant multiplier/shift per out channel (NULL on last) */
} se_layer_t;

typedef struct {
    const se_layer_t *layers;
    uint8_t n_layers;
} se_model_t;

typedef struct {
    /* front-end */
    int32_t s1, s2, s3;
    uint8_t started;
    /* event state machine */
    int16_t ring[SE_RING];
    uint32_t n;                /* index of the sample being processed; wraps mod 2^32 (~59.6 h at 20 kHz) */
    uint32_t n0;               /* index of the first sample after init (windows may not reach before it) */
    uint8_t in_ev, has_pending;
    int32_t best;
    uint32_t best_i, end, dead_until, pend_i;
    /* calibration constants (set once, from an unsupervised noise estimate) */
    int32_t gate_thr;          /* candidate if y < -gate_thr */
    int32_t gain;              /* int8 input = clamp((y*gain + 128) >> 8) */
    int32_t logit_thr;         /* spike if logit > logit_thr */
    const se_model_t *model;   /* NULL -> plain threshold detector */
} se_state_t;

typedef struct {
    uint32_t idx;              /* trough sample index */
    int32_t logit;             /* classifier logit (0 when model == NULL) */
    uint8_t accepted;
} se_event_t;

void se_init(se_state_t *s, int32_t gate_thr, int32_t gain, int32_t logit_thr, const se_model_t *m);
/* Same, but start the sample counter at n0 (used to test counter wrap-around). */
void se_init_at(se_state_t *s, int32_t gate_thr, int32_t gain, int32_t logit_thr, const se_model_t *m, uint32_t n0);
int32_t se_frontend(se_state_t *s, int16_t x);
/* Process one raw ADC sample. Returns 1 and fills *ev when a candidate is resolved. */
int se_step(se_state_t *s, int16_t x, int16_t *y_out, se_event_t *ev);
/* Reentrant: activations live on the stack (2 x SE_MAX_ACT bytes). */
int32_t se_model_run(const se_model_t *m, const int8_t *x);
#endif
