"""spike-edge: spike detection and data reduction under implant-like constraints."""
FS = 20_000          # sampling rate (Hz)
ADC_BITS = 10        # raw sample width used for the broadband bit-rate
LSB_UV = 1.0         # 1 uV per ADC count, range +-512 uV
TOL = 10             # match tolerance, samples (0.5 ms)
SEARCH = 15          # trough search after threshold crossing, samples (0.75 ms)
DEAD = 20            # dead time after a detected trough, samples (1 ms)
PRE, POST = 10, 22   # CNN window around trough: [t-PRE, t+POST)  -> 32 samples
WIN = PRE + POST
