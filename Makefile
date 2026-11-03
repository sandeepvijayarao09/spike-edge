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

YGER = https://zenodo.org/records/1205233/files
REAL_RECS = 20160426_patch3 20160426_patch2 20170622_patch1
real:        ## download ~4.7 GB of Yger et al. 2018 (CC-BY-4.0) and evaluate the frozen models
	for r in $(REAL_RECS); do d=data/yger2018_$$r; mkdir -p $$d; \
	  [ -n "$$(ls $$d/*.txt 2>/dev/null)" ] || (curl -sL --retry 3 -o $$d/$$r.tar.gz "$(YGER)/$$r.tar.gz?download=1" && tar -xzf $$d/$$r.tar.gz -C $$d && rm $$d/$$r.tar.gz); done
	$(PY) -I scripts/05_real_data.py $(addprefix data/yger2018_,$(REAL_RECS)) | tee results/real_yger2018.log
	$(PY) scripts/04_report.py
