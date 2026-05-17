# =============================================================================
# AISeism Seismic Data Processing Pipeline - Makefile
# =============================================================================
# Pipeline stages:
#   1. DATA ACQUISITION  : year, stations
#   2. PHASE PICKING     : SeisBenchPicker<YYYY>  (e.g. make SeisBenchPicker2022)
#   3. PHASE ASSOCIATION : GaMMA<YYYY>, PyOcto<YYYY>, REAL<YYYY>
#   4. EVENT LOCATION    : NonLinLoc<YYYY>, NonLinLoc3D<YYYY>
#   5. CATALOG MANAGEMENT: parser, clean
#
# USAGE:
#   make download               - Download waveforms in N-day chunks
#		make download YEAR=2025 STEP_DAYS=10 OFFSET_DAYS=0 END_DAY=365
#   make SeisBenchPicker2022  	- Run phase picker for 2022 (full year)
#   make SeisBenchPicker<YYYY>	- Run phase picker for any 4-digit year
#   make PyOcto<YYYY>.        	- Associate phases into events
# 	make GaMMA<YYYY>    		 	  - Run GaMMA associator (placeholder)
# 	make REAL<YYYY>     	  		- Run REAL associator (placeholder)
#   make NonLinLoc<YYYY>        - Run 1D event location (placeholder)
#   make NonLinLoc3D<YYYY>      - Run 3D event location (placeholder)
#   make station<YYYY>          - Process station metadata for the specified year
#   make clean                	- Remove generated catalogs
#
# DEPENDENCIES:
#   - Analyze.sh: Job submission wrapper script to activate SBC
#   - ml_catalog_run: ML catalog generation tool
#   - $(SRC_PATH)/*: OGS-specific Python scripts
#
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION VARIABLES (override at runtime with: make target VAR=value)
# -----------------------------------------------------------------------------

# General execution settings
USER ?= $(shell whoami)
ANALYZE ?= Analyze.sh

# Data range settings
YEAR ?= 2025
STEP_DAYS ?= 5
OFFSET_DAYS ?= 0
END_DAY ?= 366 # Max 366 to cover leap years; actual end date will be adjusted in the loop

# Calculate START_DATE and END_DATE based on YEAR, OFFSET_DAYS
START_DATE ?= $(shell date -d "$(YEAR)-01-01 + $(OFFSET_DAYS) days" +%Y%m%d)
END_DATE ?= $(shell date -d "$(YEAR)-12-31" +%Y%m%d)

# Validate START_DATE and END_DATE are valid dates
_START_DATE_CHECK := $(shell date -d "$(START_DATE)" +%Y%m%d 2>/dev/null || echo "invalid")
_END_DATE_CHECK := $(shell date -d "$(END_DATE)" +%Y%m%d 2>/dev/null || echo "invalid")

$(if $(filter invalid,$(_START_DATE_CHECK)),$(error Invalid START_DATE: $(START_DATE)))
$(if $(filter invalid,$(_END_DATE_CHECK)),$(error Invalid END_DATE: $(END_DATE)))

# Paths
WORK_ROOT ?= /leonardo_work/IscrC_AISeism/WORK
SRC_PATH ?= /leonardo_work/IscrC_AISeism/PhD/OGS/src
WAVE_PATH ?= $(WORK_ROOT)/waveform
MANUAL_PATH ?= $(WORK_ROOT)/manual/OnlyEqNLL1D
CONDA_PATH ?= /leonardo_work/IscrC_AISeism/.miniconda3/bin/conda
PYTHON_PATH ?= /leonardo_work/IscrC_AISeism/.miniconda3/envs/SBC_3.12/bin/python
STATION_PATH ?= $(WAVE_PATH)/station

# Output catalog directory paths on the shared filesystem
# OGS catalog: Primary output from SeisBench phase picker
ogsCatalog ?= $(WORK_ROOT)/catalogs/OGS$(YEAR)
# Picker model catalog: Output from SeisBench picker (used as input for associators)
pickCatalog ?= $(ogsCatalog)/SeisBenchPicker

# PyOcto catalog: Output from PyOcto phase associator
pyoctoCatalog ?= $(WORK_ROOT)/catalogs/PyOcto$(YEAR)
# NonLinLoc catalog: Output from NonLinLoc 3D locator
nonlinlocCatalog ?= $(WORK_ROOT)/catalogs/NonLinLoc$(YEAR)

# Picker settings
PICK_OUTPUT ?= $(ogsCatalog)$(YEAR)

help:
	@echo "Available targets:"
	@echo "  make download YEAR=2025 STEP_DAYS=5 START_DAY=160"
	@echo "  make compress YEAR=2025 STEP_DAYS=5 START_DAY=160"
	@echo "  make decompress YEAR=2025 STEP_DAYS=5 START_DAY=160"
	@echo "  make PyOctoAssociator"
	@echo "  make GaMMAAssociator"
	@echo "  make REALAssociator"
	@echo "  make NonLinLoc"
	@echo "  make NonLinLoc3D"
	@echo "  make station YEAR=2025"
	@echo "  make parser YEAR=2025"

setup:
	@echo "Setting up environment and dependencies..."
	# Add any setup commands here (e.g., module loads, environment variables)

	@echo "Setup complete."

# =============================================================================
# DATA ACQUISITION
# =============================================================================

# Download waveforms for $(YEAR) in $(STEP_DAYS)-day segments.
# Adjust seq range (0 to 365) or $(STEP_DAYS) to change coverage/chunk size.
download:
	@for i in $$(seq $(OFFSET_DAYS) $(STEP_DAYS) $(END_DAY)); do \
		start=$$(date -d "$(YEAR)-01-01 + $$i days" +%Y%m%d); \
		end=$$(date -d "$(YEAR)-01-01 + $$((i+$(STEP_DAYS))) days" +%Y%m%d); \
		[ "$$(date -d "$(YEAR)-01-01 + $$i days" +%Y)" = "$(YEAR)" ] || continue; \
		[ "$$(date -d "$(YEAR)-01-01 + $$((i+$(STEP_DAYS))) days" +%Y)" = "$(YEAR)" ] || end=$(YEAR)1231; \
		echo "Downloading waveforms from $$start to $$end (OFFSET_DAYS $$i days)"; \
		bash $(ANALYZE) download 1 1 "DownLoad $$start $$end" \
			$(PYTHON_PATH) $(SRC_PATH)/ogsdownloader.py -D $$start $$end -d $(WAVE_PATH); \
	done

compress:
	bash $(ANALYZE) compress 1 1 Compressor \
		$(PYTHON_PATH) $(WORK_ROOT)/compressor.py -C $(WAVE_PATH) . -v

decompress:
	bash $(ANALYZE) compress 1 1 Decompressor \
		$(PYTHON_PATH) $(WORK_ROOT)/compressor.py -D $(WAVE_PATH)/$(YEAR) \
			$(WAVE_PATH) -v && find $(WAVE_PATH)/$(YEAR)/* -type f -name "*.tar.gz" -print -delete

retrieve:
	@echo "Retrieving waveforms for $(YEAR) to local scratch space (/leonardo_scratch/large/userexternal/$(USER)/)..."
	rsync -avh /leonardo_store/DRES_EchoArch/amagrin0/Ken/$(YEAR) \
		/leonardo_scratch/large/userexternal/$(USER)/

# =============================================================================
# PHASE PICKING — pattern rule handles any 4-digit year suffix (YYYY)
# Usage: make SeisBenchPicker2022 | make SeisBenchPicker2024 | etc.
# =============================================================================
SeisBenchPicker%:
	bash $(ANALYZE) $(USER) 1 1 $@ \
		ml_catalog_run +libpath=. \
		output_path=$(ogsCatalog)$* \
		data.starttime=$*-01-01 \
		data.endtime=$*-12-31

# =============================================================================
# PHASE ASSOCIATION
# =============================================================================
PyOcto%:
	# Copy SeisBench picker results to PyOcto working directory
	cp -r $(ogsCatalog)/SeisBenchPicker $(pyoctoCatalog)
	# Run PyOcto associator via Analyze.sh job submission
	bash $(ANALYZE) $(USER) 1 1 $@ \
		ml_catalog_run +libpath=. \
		output_path=$(pyoctoCatalog) \
		group_modules/associator=ogspyocto.yaml

GaMMAAssociator:


REALAssociator:
	# Placeholder for REALAssociator job submission and execution
	@echo "Running REALAssociator for $(YEAR)..."
	# Add commands to run REALAssociator here (e.g., via Analyze.sh or directly)

# =============================================================================
# EVENT LOCATION
# =============================================================================
NonLinLoc1D:
	# Placeholder for NonLinLoc job submission and execution
	@echo "Running NonLinLoc event location for $(YEAR)..."
	# Add commands to run NonLinLoc here (e.g., via Analyze.sh or directly)


NonLinLoc3D:
	# Placeholder for NonLinLoc job submission and execution
	@echo "Running NonLinLoc 3D event location for $(YEAR)..."
	# Add commands to run NonLinLoc 3D here (e.g., via Analyze.sh or directly)

# =============================================================================
# PhD
# =============================================================================
PhD:
	@echo "Running PhD-specific analysis for $(YEAR)..."
	# Add any PhD-specific targets or commands here (e.g., custom plots, metrics)



# =============================================================================
# CATALOG MANAGEMENT
# =============================================================================

# TARGET: station
# Downloads and processes station metadata for the specified year.
# Retrieves station information (coordinates, response, etc.) from OGS
# and stores it in the 'station' directory. Uses waveform directory
# to cross-reference available data.
station:
	@echo "Processing station metadata for waveforms from $(START_DATE) to $(END_DATE)..."; \
	bash $(ANALYZE) $(USER) 1 1 "Stations $(START_DATE) - $(END_DATE)" \
		$(PYTHON_PATH) $(SRC_PATH)/ogsstation.py $(SRC_PATH) $(WAVE_PATH) $(STATION_PATH) $(START_DATE) $(END_DATE) $(WORK_ROOT)/catalogs/OGS$(YEAR)


# TARGET: parser
# Parses and merges manual earthquake catalogs from OGS.
# Reads manual pick files from the shared filesystem and consolidates
# them into a unified format. The --merge flag combines multiple files.
parser:
	$(PYTHON_PATH) $(SRC_PATH)/ogsparser.py -f $(MANUAL_PATH)/* --merge \
		-o $(WORK_ROOT)/catalogs/$(notdir $(patsubst %/,%,$(MANUAL_PATH)))

# WARNING: deletes all picker/associator outputs under catalog/OGS
clean:
	rm -rf $(ogsCatalog)

.PHONY: help setup download compress decompress year station PyOctoAssociator parser clean
