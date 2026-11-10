PY ?= .venv/bin/python

.PHONY: all detection reduction embedded report test c
all: detection reduction embedded report  # `make real` adds the real-data check (4.7 GB download)

detection:   ## generate data, baselines, train + quantize models (~8 min on an M-series Mac)
	$(PY) scripts/01_detection.py | tee results/detection.log
reduction:   ## bits/s vs velocity decoding, plus the untuned-hash control (~3 min)
	$(PY) scripts/02_reduction.py | tee results/reduction.log
	$(PY) scripts/02_reduction.py --untuned-hash | tee results/reduction_untuned.log
embedded:    ## C bit-exactness, host latency, Cortex-M4 code size, cycle/power estimate
	$(PY) scripts/03_embedded.py | tee results/embedded.log
report:      ## figures + README tables from results/*.json
	$(PY) scripts/04_report.py
test:
	$(PY) -m pytest -q
c:           ## build the host harness for every committed model
	$(PY) -c "from spike_edge import qnn; from spike_edge.export_c import write_header, build; import glob; [build(write_header(*qnn.load(p)[:2]).stem[6:]) for p in glob.glob('models/*.npz')]"

real:        ## all 19 Yger 2018 recordings (~31 GB, CC-BY-4.0) + 7 Marques-Smith cortical cells (streamed, CC BY-SA 4.0)
	$(PY) -I scripts/fetch_yger.py -j 3
	$(PY) -I scripts/fetch_marques_smith.py
	$(PY) -I scripts/05_real_data.py | tee results/real_paired_gt.log
	$(PY) scripts/04_report.py
