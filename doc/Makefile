PDFLATEX ?= pdflatex
LATEXMK  ?= latexmk
BIBTEX   ?= bibtex
UNITS_DIR := UNITS
UNITS_MAIN := UNITS.tex
BUILD_DIR := UNITS
JOB_NAME := UNITS
EDUAI_ABSTRACT_DIR := EduAI/abstract
EDUAI_ABSTRACT_BUILD_DIR := $(EDUAI_ABSTRACT_DIR)/build
EDUAI_SLIDES_DIR := EduAI/slides
EDUAI_SLIDES_BUILD_DIR := $(EDUAI_SLIDES_DIR)/build

.PHONY: all units units-fast units-check units-release eduai-abstract eduai-slides clean

all: units

# 1. Default smart incremental build: uses latexmk for minimal necessary passes
units:
	mkdir -p $(BUILD_DIR)
	cd $(UNITS_DIR) && $(LATEXMK) -pdf -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error $(UNITS_MAIN)

# 2. Fast single-pass preview (for quick layout inspection)
units-fast:
	mkdir -p $(BUILD_DIR)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error $(UNITS_MAIN)

# 3. Fast syntax/macro/citation validation check (no PDF disk write, ~15s)
units-check:
	mkdir -p $(BUILD_DIR)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -draftmode -interaction=nonstopmode -halt-on-error $(UNITS_MAIN)

# 4. Full 3-pass legacy release build
units-release:
	mkdir -p $(BUILD_DIR)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)
	cd $(BUILD_DIR) && BIBINPUTS=../../$(UNITS_DIR): $(BIBTEX) $(JOB_NAME)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)

# Build the EduAI conference abstract without placing auxiliary files beside the source.
eduai-abstract:
	mkdir -p $(EDUAI_ABSTRACT_BUILD_DIR)
	cd $(EDUAI_ABSTRACT_DIR) && $(LATEXMK) -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex

# Build the EduAI conference slides without placing auxiliary files beside the source.
eduai-slides:
	mkdir -p $(EDUAI_SLIDES_BUILD_DIR)
	cd $(EDUAI_SLIDES_DIR) && $(LATEXMK) -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex

# 5. Clean auxiliary build files
clean:
	cd $(UNITS_DIR) && $(LATEXMK) -c -jobname=$(JOB_NAME)