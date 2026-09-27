PDFLATEX ?= pdflatex
LATEXMK  ?= latexmk
BIBTEX   ?= bibtex
UNITS_DIR := UNITS
UNITS_MAIN := UNITS.tex
BUILD_DIR := UNITS
JOB_NAME := UNITS

.PHONY: all units units-fast units-check units-release clean

all: units

# 1. Default smart incremental build: uses latexmk for minimal necessary passes
units:
	mkdir -p $(BUILD_DIR)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)
	cd $(BUILD_DIR) && BIBINPUTS=../../$(UNITS_DIR): $(BIBTEX) $(JOB_NAME)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)
	cd $(UNITS_DIR) && $(PDFLATEX) -jobname=$(JOB_NAME) -interaction=nonstopmode -halt-on-error -output-directory=../$(BUILD_DIR) $(UNITS_MAIN)