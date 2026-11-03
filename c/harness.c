/* Host harness: stream a recording through the C pipeline and dump outputs.
 * in : int32 n, gate_thr, gain, logit_thr ; int16 raw[n]
 * out: int16 y[n] ; int32 n_ev ; n_ev x (uint32 idx, int32 logit, int32 accepted)
 * flags: --no-model (threshold only), --start N (start the sample counter at N) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spike_edge.h"
#include MODEL_HEADER

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.bin out.bin [--no-model] [--start N]\n", argv[0]); return 2; }
    int use_model = 1; uint32_t n0 = 0;
    for (int a = 3; a < argc; a++) {
        if (!strcmp(argv[a], "--no-model")) use_model = 0;
        else if (!strcmp(argv[a], "--start") && a + 1 < argc) n0 = (uint32_t)strtoul(argv[++a], 0, 10);
    }
    FILE *f = fopen(argv[1], "rb"); if (!f) return 1;
    int32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4) return 1;
    int16_t *raw = malloc(sizeof(int16_t) * hdr[0]), *y = malloc(sizeof(int16_t) * hdr[0]);
    if (fread(raw, 2, hdr[0], f) != (size_t)hdr[0]) return 1;
    fclose(f);
    static se_state_t s;
    se_init_at(&s, hdr[1], hdr[2], hdr[3], use_model ? &SE_MODEL : NULL, n0);
    int32_t *evs = malloc(sizeof(int32_t) * 3 * (hdr[0] / SE_DEAD + 1)); int32_t n_ev = 0;
    for (int32_t i = 0; i < hdr[0]; i++) {
        se_event_t ev;
        if (se_step(&s, raw[i], &y[i], &ev)) {
            evs[3 * n_ev] = (int32_t)ev.idx; evs[3 * n_ev + 1] = ev.logit; evs[3 * n_ev + 2] = ev.accepted; n_ev++;
        }
    }
    FILE *o = fopen(argv[2], "wb");
    fwrite(y, 2, hdr[0], o); fwrite(&n_ev, 4, 1, o); fwrite(evs, 4, 3 * n_ev, o);
    fclose(o);
    printf("samples %d events %d sizeof(state) %zu\n", hdr[0], n_ev, sizeof(se_state_t));
    return 0;
}
