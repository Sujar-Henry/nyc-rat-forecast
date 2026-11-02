# PYTHONPATH=src keeps the pipeline runnable without relying on the editable
# install's .pth file (some macOS setups mark files inside .venv as hidden,
# and Python 3.12+ skips hidden .pth files).
PY  ?= .venv/bin/python
RUN  = PYTHONPATH=src $(PY) -m ratcast

.PHONY: setup download clean features train evaluate maps site deploy all test

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]"

download:   ; $(RUN) download
clean:      ; $(RUN) clean
features:   ; $(RUN) features
train:      ; $(RUN) train
evaluate:   ; $(RUN) evaluate
maps:       ; $(RUN) maps
site:       ; $(RUN) site
all:        ; $(RUN) all

test:
	PYTHONPATH=src $(PY) -m pytest -q

# site/ is the published GitHub Pages repo; this pushes a rebuilt page.
deploy: site
	cd site && git add -A && (git diff --cached --quiet || git commit -m "Update map") && git push
