# =============================================================================
# AISeism Seismic Data Processing Pipeline for HPC workflow - Makefile
# =============================================================================
# Pipeline stages:
#   1. DATA ACQUISITION   : download, compress, decompress, retrieve, station
#   2. PHASE PICKING      : PhaseNet / EQTransformer with dataset+threshold args
#   3. PHASE ASSOCIATION  : PyOcto / GaMMA / REAL with picker-input args
#   4. EVENT LOCATION     : NLL1D / NLL3D with associator+picker-input args
#   5. CATALOG MANAGEMENT : parse, clean
#
# QUICK START / USAGE EXAMPLES:
#   make help
#   make init [DRY_RUN=1]   # Initialize the WORK directory structure
#   make download YEAR=2025 STEP_DAYS=5 OFFSET=0 END_DAY=365
#   make download YEAR=2025 STEP_DAYS=5 OFFSET=0 END_DAY=365 THREADS=4
#	  make compress           # Compress raw waveforms into yearly tar.gz files
#	  make decompress         # Decompress yearly tar.gz files into raw waveforms
#   make retrieve           # rsync waveforms to scratch for manual analysis
#   make index              # Build/update Squirrel indexes for the waveform archive
#   make PyOcto             # Run PyOcto with the default picker input
#   make GaMMA              # Run GaMMA with the default picker input
#   make REAL               # Run REAL with the default picker input
#   make NLL1D              # Run NonLinLoc 1D with the default assoc+picking inputs
#   make NLL3D              # Run NonLinLoc 3D with the default assoc+picking inputs
#   make base               # Just make the base catalog for quick iteration
#   make station YEAR=2025
#   make parse
#   make clean              # WARNING: deletes generated catalogs
#   make MHPC               # Submit all registered models/assocs to SLURM for the given date range
#   make PhD                # Submit all registered models/assocs to SLURM for the given date range
#   make test               # Run a quick test of the ml_catalog_run entrypoint
#
# Bracketed targets must be quoted because square brackets are shell glob characters:
#   make PhaseNet[INSTANCE,0.2]
#   make "PyOcto[PhaseNet,INSTANCE,0.2]"
#   make "NLL1D[PyOcto,PhaseNet,INSTANCE,0.2]" START_DATE=2024-03-20 END_DATE=2024-06-20
# Use `make -n TARGET ...` to preview a command.
#
# Database/Picker/associator/NLL targets are generated from the registered
# stage-name lists below via the
# _INDEXER_RULES / _PICKER_RULES / _ASSOC_RULES / _NLL_RULES macros.
#
# DEPENDENCIES:
#   - LAUNCHME.sh           : SLURM job-submission wrapper (activates SBC env)
#   - $(SBC_RUN_BIN)        : ML catalog generation entrypoint (Hydra) [= ml_catalog_run]
#   - $(OGS_PATH)/src/*.py  : OGS-specific Python scripts
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION (override at runtime: `make target VAR=value`)
# -----------------------------------------------------------------------------

# General
USER        ?= $(shell whoami)
PROJECT     ?= Leonardo_Dev
THREADS     ?= 4
CONDA_ENV   ?= SBC_3.12

# Date Variables
YEAR        ?= 2026
STEP_DAYS   ?= 5
OFFSET      ?= 0
END_DAY     ?= 366    # max 366 (leap years); loop clamps to year end

# Paths
WORK_PATH   		?= /leonardo_work/IscrC_AISeism/WORK
OGS_PATH    		?= $(abspath $(WORK_PATH)/../PhD/OGS)
NLL_PATH    		?= $(abspath $(WORK_PATH)/../NonLinLoc)
SBC_PATH				?= $(abspath $(WORK_PATH)/../ml_catalog_main)
CONDA_INSTALLER	?= $(abspath $(WORK_PATH)/../Miniconda3-latest-Linux-x86_64.sh)
CONDA_ROOT 			?= $(abspath $(WORK_PATH)/../.miniconda3)
CATALOG_PATH 		?= $(abspath $(WORK_PATH)/catalog)
DATASET_PATH 		?= $(abspath $(WORK_PATH)/dataset)
STATION_PATH 		?= $(abspath $(WORK_PATH)/station)
WAVE_PATH   		?= $(abspath $(WORK_PATH)/waveform)

START_DATE  ?= $(shell date -d "$(YEAR)-01-01 + $(OFFSET) days" +%Y%m%d)
END_DATE    ?= $(shell date -d "$(YEAR)-12-31" +%Y%m%d)

_START_DATE_CHECK := $(shell date -d "$(START_DATE)" +%Y%m%d 2>/dev/null || echo invalid)
_END_DATE_CHECK   := $(shell date -d "$(END_DATE)"   +%Y%m%d 2>/dev/null || echo invalid)
$(if $(filter invalid,$(_START_DATE_CHECK)),$(error Invalid START_DATE: $(START_DATE)))
$(if $(filter invalid,$(_END_DATE_CHECK)),$(error Invalid END_DATE: $(END_DATE)))

ifeq ($(YEAR),$(shell date -d "$(START_DATE)" +%Y))
else
  YEAR := $(shell date -d "$(START_DATE)" +%Y)
endif

# More paths
TT_DB        	?= $(OGS_PATH)/data/VelocityModel/1D/tt_db.txt

# Executable binaries
CONDA_BIN			?= $(CONDA_ROOT)/bin/conda
PYTHON_BIN		?= $(CONDA_ROOT)/envs/$(CONDA_ENV)/bin/python
SBC_RUN_BIN		?= ml_catalog_run +libpath=.

# SLURM template (defaults to user template if present, else default)
SLURM_TEMPLATE ?= $(if $(wildcard $(OGS_PATH)/utils/Leonardo/$(USER).sh),$(USER),default)

# Default picker/associator roots that chained stages can reuse.
base        := OGS/$(YEAR)
BASE_PICKER := $(WORK_PATH)/catalogs/$(base)/SeisBenchPicker
BASE_ASSOC  := $(WORK_PATH)/catalogs/$(base)/PyOcto

# Default selections and registered stage names for generated runtime targets.
# The generated rules inject these values into the Hydra overrides.
INDEXER     ?= Indexer
INDEXERS		:= Indexer
MODEL 			?= PhaseNet
MODELS			:= PhaseNet EQTransformer
DATASET 		?= INSTANCE
DATASETS		:= Original INSTANCE STEAD SCEDC
THRESHOLD		?= 0.1
THRESHOLDS  := 0.1 0.2 0.3
ASSOC			  ?= PyOcto
ASSOCS			:= PyOcto GaMMA
NLL				  ?= NLL1D
NLLS				:= NLL1D NLL3D

# -----------------------------------------------------------------------------
# PHONY DECLARATIONS
# -----------------------------------------------------------------------------
.DEFAULT_GOAL := help

.PHONY: help setup all init base MHPC parse test clean \
  download compress decompress retrieve station index \
  $(MODELS) $(ASSOCS) $(NLLS)

# -----------------------------------------------------------------------------
# DEFAULT / META
# -----------------------------------------------------------------------------
all: base # 2
	@$(MAKE) --no-print-directory '$(NLL)[$(ASSOC),$(MODEL),$(DATASET),$(THRESHOLD)]'

help: #
	@echo "AISeism Makefile — common targets:"
	@echo "  make init"
	@echo "  make download   YEAR=YYYY STEP_DAYS=N OFFSET=D END_DAY=D"
	@echo "  make download   YEAR=YYYY STEP_DAYS=N OFFSET=D END_DAY=D THREADS=4"
	@echo "  make compress | decompress | retrieve   YEAR=YYYY 	(deprecated; retained for compatibility)"
	@echo "  make station    YEAR=YYYY"
	@echo "  make index THREADS=8 START_DATE=1977-01-01 END_DATE=2025-12-31"
	@echo "  make PhaseNet | EQTransformer | ...	(picker models with default dataset/threshold)"
	@echo "  make PhaseNet[INSTANCE,0.1] | EQTransformer[STEAD,0.2] | ...	(picker models with dataset/threshold overrides)"
	@echo "  make PyOcto | GaMMA | REAL           (associator with default picker input)"
	@echo "  make PyOcto[PhaseNet,INSTANCE,0.1] | GaMMA[EQTransformer,STEAD,0.2] | REAL[PhaseNet,SCEDC,0.3] ...	(associator with picker overrides)"
	@echo "  make NLL1D | NLL3D                   (locator with default picker+assoc input)"
	@echo "  make NLL1D[PyOcto,PhaseNet,INSTANCE,0.1] | NLL3D[GaMMA,EQTransformer,STEAD,0.2] | ...	(locator with picker+assoc overrides)"
	@echo "  make base                            (just make the base catalog for quick iteration on later stages)"
	@echo "  make parse | clean"
	@echo "  make MHPC                            (submit all registered models/assocs to SLURM for the given date range)"
	@echo "  make PhD                             (submit all registered models/assocs to SLURM for the given date range)"
	@echo "  make test                            (run a quick test of the SBC_RUN_BIN entrypoint)"
	@echo "  make help                            (this message)"

setup:
	@echo "Setup placeholder — module loads / env activation go here."

# =============================================================================
# INITIALIZATION
# =============================================================================
init:
	@WORK_PATH="$(WORK_PATH)" OGS_PATH="$(OGS_PATH)" \
		NLL_PATH="$(NLL_PATH)" DATASET_PATH="$(DATASET_PATH)" \
		CONDA_ROOT="$(CONDA_ROOT)" CONDA_INSTALLER="$(CONDA_INSTALLER)" \
		CONDA_ENV="$(CONDA_ENV)" SBC_PATH="$(SBC_PATH)" \
		bash "$(OGS_PATH)/utils/Leonardo/init.sh"

# =============================================================================
# DATA ACQUISITION
# =============================================================================

# Download waveforms in $(STEP_DAYS)-day chunks, clamped to $(YEAR).
download: #
	[[ "$(STEP_DAYS)" =~ ^[1-9][0-9]*$$ ]] || {
		printf 'ERROR: STEP_DAYS must be a positive integer: %s\n' "$(STEP_DAYS)" >&2
		exit 2
	}
	[[ "$(OFFSET)" =~ ^[0-9]+$$ ]] || {
		printf 'ERROR: OFFSET must be a non-negative integer: %s\n' "$(OFFSET)" >&2
		exit 2
	}
	[[ "$(END_DAY)" =~ ^[1-9][0-9]*$$ ]] || {
		printf 'ERROR: END_DAY must be a positive integer: %s\n' "$(END_DAY)" >&2
		exit 2
	}
	for i in $$(seq $(OFFSET) $(STEP_DAYS) $$(END_DAY)); do \
	  start=$$(date -d "$(YEAR)-01-01 + $$i days" +%Y%m%d); \
	  [ "$$(date -d "$$start" +%Y)" = "$(YEAR)" ] || continue; \
	  end=$$(date -d "$$start + $$STEP_DAYS - 1 days" +%Y%m%d); \
	  [ "$$end" -le "$$(END_DAY)" ] || end=$$(END_DAY); \
	  [ "$$(date -d "$$end" +%Y)" = "$(YEAR)" ] || end=$(YEAR)1231; \
	  echo "Downloading waveforms $$start → $$end (offset $$i days)"; \
	  bash LAUNCHME.sh download 1 1 "DownLoad_$${start}_$${end}" \
	    $(PYTHON_BIN) $(OGS_PATH)/src/ogsdownloader.py \
				-D $$start $$end \
				-d $(WAVE_PATH) \
				--threads $(THREADS) \
				-S "* -SP -OL -ED"; \
	done

compress: #
	@echo "WARNING: make compress is deprecated; use the current waveform archive workflow instead." >&2
	bash LAUNCHME.sh compress 1 1 Compressor \
	  $(PYTHON_BIN) $(WORK_PATH)/compressor.py \
			-C $(WAVE_PATH) . -v

decompress:
	@echo "WARNING: make decompress is deprecated; use the current waveform archive workflow instead." >&2
	bash LAUNCHME.sh compress 1 1 Decompressor \
	  $(PYTHON_BIN) $(WORK_PATH)/compressor.py \
			-D $(WAVE_PATH)/$(YEAR) \
	    $(WAVE_PATH) -v && \
	find $(WAVE_PATH)/$(YEAR)/* -type f -name "*.tar.gz" -print -delete

retrieve:
	@echo "WARNING: make retrieve is deprecated; use the current waveform archive workflow instead." >&2
	@echo "Retrieving YEAR=$(YEAR) waveforms to /leonardo_scratch/large/userexternal/$(USER)/"
	rsync -avh /leonardo_store/DRES_EchoArch/amagrin0/Ken/$(YEAR) \
	  /leonardo_scratch/large/userexternal/$(USER)/

# Station metadata for the configured date range.
station:
	@echo "Processing station metadata $(START_DATE) → $(END_DATE)..."
	bash LAUNCHME.sh $(SLURM_TEMPLATE) 1 1 "Stations $(START_DATE)-$(END_DATE)" \
	  $(PYTHON_BIN) $(OGS_PATH)/src/ogsstation.py \
	    $(OGS_PATH)/src $(WAVE_PATH) $(STATION_PATH) \
	    $(START_DATE) $(END_DATE) $(WORK_PATH)/catalogs/$(base)

# Build/update the sharded Squirrel indexes for the full waveform archive.
index:
	$(SBC_RUN_BIN) +index_only=true data=ogsDB data.check=true \
	  data.index_workers=$(THREADS) \
	  data.starttime=$(START_DATE) \
	  data.endtime=$(END_DATE)

# =============================================================================
# INDEXER / PICKER / ASSOCIATOR / NLL — RUNTIME PARAMETRIC RULES
# =============================================================================
#
# Instead of enumerating every combination in spec tables, we generate one
# pattern rule per known stage name (from $(MODELS) / $(ASSOCS) / $(NLLS))
# and parse the bracketed argument list at recipe time.
#
# USAGE EXAMPLES:
#   make <Index>   																# runs the indexer stage (no args)
#   make <Model>                                  # uses $(DATASET),$(THRESHOLD)
#   make '<Model>[<Dataset>,<Threshold>]'
#   make <Assoc>                                  # chains off $(BASE_PICKER)
#   make '<Assoc>[<Model>,<Dataset>,<Threshold>]'
#   make <NLL>                                    # chains off $(BASE_ASSOC)
#   make '<NLL>[<Assoc>,<Model>,<Dataset>,<Threshold>]'
#
# NOTE: bracketed forms must be quoted on the shell so that `[`,`,`,`]` are
#       not glob-expanded.
#
# YAML mapping is centralised in the *_YAML_<NAME> variables below; add a new
# entry there and a new word to $(MODELS)/$(ASSOCS)/$(NLLS) to register a
# stage with no further edits.
# =============================================================================

# --- YAML mapping (override one line to point a stage at a different config) -
INDEXER                ?= Indexer
INDEXER_YAML           ?= ogsdata.yaml
PICKER                 ?= SeisBenchPicker
PICKER_YAML            ?= ogsseisbenchpicker.yaml
PICKER_YAML_PhaseNet      ?= $(PICKER_YAML)
PICKER_YAML_EQTransformer ?= $(PICKER_YAML)

ASSOC                  ?= OGSPickStatQC
ASSOC_YAML             ?= ogspyocto.yaml
ASSOC_YAML_PyOcto      ?= $(ASSOC_YAML)
ASSOC_YAML_GaMMA       ?= ogsgamma.yaml
ASSOC_YAML_REAL        ?= ogsreal.yaml

LOCATOR                ?= OGSLocalMagnitude
LOCATOR_YAML           ?= ogsnonlinloc1d.yaml
LOCATOR_YAML_NLL1D     ?= $(LOCATOR_YAML)
LOCATOR_YAML_NLL3D     ?= ogsnonlinloc3d.yaml

# Reserved mapping for associator output folder names. The current locator
# rules still symlink `OGSPickStatQC` directly.
ASSOC_DIR_PyOcto       ?= PyOcto
ASSOC_DIR_GaMMA        ?= GaMMA
ASSOC_DIR_REAL         ?= REAL

# =============================================================================
# RULE GENERATORS
# =============================================================================
# Each generator emits two rules per stage name:
#   1. A default rule with no `[...]`
#   2. A pattern rule `<Name>[%]:` whose stem `$*` is parsed in the shell
#
# Escaping notes for `$(eval $(call _MACRO,...))`:
#   $$X   → literal $X for Make (expanded once `eval` parses the rule)
#   $$$$x → literal $x for the shell (Make consumes one $)
#   $$*   → Make's $* automatic var, expanded when the rule fires
# =============================================================================

# --- INDEXER -------------------------------------------------------------------
define _INDEXER_RULES

$1:
	bash LAUNCHME.sh $$(USER) 1 1 "Indexer $$(YEAR)" \
	  $$(SBC_RUN_BIN) +index_only=true data=ogsDB data.check=true \
	  data.index_workers=$$(THREADS) \
	  data.starttime=$$(START_DATE) data.endtime=$$(END_DATE)
endef

# --- PICKER ------------------------------------------------------------------
# id convention: <Model>[<Dataset>,<Threshold>]
define _PICKER_RULES
$1: ; @$$(MAKE) --no-print-directory '$1[$$(DATASET),$$(THRESHOLD)]'

$1[%]:
	@stem='$$*'; dataset="$$$${stem%%,*}"; threshold="$$$${stem##*,}"; \
	id="$1[$$$$dataset,$$$$threshold]"; \
	picker_id="$$$$dataset/$$$$threshold/$$(YEAR)"; \
	mkdir -p "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$picker_id/SeisBenchPicker"; \
	bash LAUNCHME.sh $$(SLURM_TEMPLATE) 1 1 "$$$$id" \
	  $$(SBC_RUN_BIN) output_path="catalogs/$$(PROJECT)/$1/$$$$picker_id" \
	  group_modules/picker=$(PICKER_YAML_$1) \
	  group_modules.picker.model._target_="seisbench.models.$1.from_pretrained" \
	  group_modules.picker.model.name=$$$${dataset,,} \
	  group_modules.picker.classify_args.P_threshold=$$$$threshold \
	  group_modules.picker.classify_args.S_threshold=$$$$threshold \
	  data.starttime=$$(START_DATE) data.endtime=$$(END_DATE)
endef

# --- ASSOCIATOR --------------------------------------------------------------
# id convention: <Assoc>[<Model>,<Dataset>,<Threshold>]
# Symlinks the upstream picker's SeisBenchPicker output dir into the assoc dir.
define _ASSOC_RULES
$1: ; @$$(MAKE) --no-print-directory '$1[$$(MODEL),$$(DATASET),$$(THRESHOLD)]'

$1[%]:
	@stem='$$*'; \
	model="$$$${stem%%,*}"; rest="$$$${stem#*,}"; \
	dataset="$$$${rest%%,*}"; threshold="$$$${rest##*,}"; \
	id="$1[$$$$model,$$$$dataset,$$$$threshold]"; \
	picker_id="$$$$model/$$$$dataset/$$$$threshold/$$(YEAR)"; \
	mkdir -p "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$picker_id"; \
	mkdir -p "$$(WORK_PATH)/catalogs/$$(PROJECT)/$$$$picker_id"; \
	ln -sfn "$$(WORK_PATH)/catalogs/$$(PROJECT)/$$$$picker_id/SeisBenchPicker" \
	  "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$picker_id/SeisBenchPicker"; \
	bash LAUNCHME.sh $$(SLURM_TEMPLATE) 1 1 "$$$$id" \
	  $$(SBC_RUN_BIN) output_path="catalogs/$$(PROJECT)/$1/$$$$picker_id" \
	  group_modules/associator=$(ASSOC_YAML_$1) \
	  data.starttime=$$(START_DATE) data.endtime=$$(END_DATE)
endef

# --- NONLINLOC ---------------------------------------------------------------
# id convention: <NLL>[<Assoc>,<Model>,<Dataset>,<Threshold>]
# Symlinks the upstream picker and associator output dirs.
define _NLL_RULES
$1: ; @$$(MAKE) --no-print-directory '$1[$$(ASSOC),$$(MODEL),$$(DATASET),$$(THRESHOLD)]'

$1[%]:
	@stem='$$*'; \
	assoc_="$$$${stem%%,*}"; rest="$$$${stem#*,}"; \
	model_="$$$${rest%%,*}"; rest="$$$${rest#*,}"; \
	dataset_="$$$${rest%%,*}"; threshold_="$$$${rest##*,}"; \
	nll_id="$1[$$$$assoc_,$$$$model_,$$$$dataset_,$$$$threshold_]"; \
	picker_id="$$$$model_/$$$$dataset_/$$$$threshold_/$$(YEAR)"; \
	assoc_id="$$$$assoc_/$$$$picker_id"; \
	mkdir -p "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$assoc_id"; \
	mkdir -p "$$(WORK_PATH)/catalogs/$$(PROJECT)/$$$$assoc_id/OGSPickStatQC"; \
	ln -sfn "$$(WORK_PATH)/catalogs/$$(PROJECT)/$$$$assoc_id/OGSPickStatQC" \
	  "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$assoc_id/OGSPickStatQC"; \
	ln -sfn "$$(WORK_PATH)/catalogs/$$(PROJECT)/$$$$picker_id/SeisBenchPicker" \
	  "$$(WORK_PATH)/catalogs/$$(PROJECT)/$1/$$$$assoc_id/SeisBenchPicker"; \
	bash LAUNCHME.sh $$(SLURM_TEMPLATE) 1 1 "$$$$nll_id" \
	  $$(SBC_RUN_BIN) output_path="catalogs/$$(PROJECT)/$1/$$$$assoc_id" \
	  group_modules/nonlinloc=$(LOCATOR_YAML_$1) \
	  data.starttime=$$(START_DATE) data.endtime=$$(END_DATE)
endef

# Instantiate rules — one pair per registered stage name.
$(foreach i,$(INDEXERS),$(eval $(call _INDEXER_RULES,$i)))
$(foreach m,$(MODELS),$(eval $(call _PICKER_RULES,$m)))
$(foreach a,$(ASSOCS),$(eval $(call _ASSOC_RULES,$a)))
$(foreach n,$(NLLS),  $(eval $(call _NLL_RULES,$n)))

# -----------------------------------------------------------------------------
# BASE catalog — single quick-iteration run reused as input by chained stages.
# -----------------------------------------------------------------------------
base:
	mkdir -p "$(WORK_PATH)/catalogs/$(base)/$(YEAR)"

MHPC:
	@$(foreach m,$(MODELS),$(foreach d,$(DATASETS),$(foreach t,$(THRESHOLDS), \
	  echo "Submitting $m[$d,$t] to SLURM..."; \
	  $(MAKE) --no-print-directory '$m[$d,$t]' PROJECT=MHPC \
	    START_DATE=20240320 END_DATE=20240620 ; )))
	@$(foreach a,$(ASSOCS),$(foreach t,$(THRESHOLDS), \
	  echo "Submitting NLL1D[$a,PhaseNet,INSTANCE,$t] to SLURM..."; \
	  $(MAKE) --no-print-directory 'NLL1D[$a,PhaseNet,INSTANCE,$t]' \
	    PROJECT=MHPC \
	    START_DATE=20240320 END_DATE=20240620 ; ))

PhD:
	@$(foreach m,$(MODELS),$(foreach d,$(DATASETS),$(foreach t,$(THRESHOLDS), \
	  echo "Submitting $m[$d,$t] to SLURM..."; \
	  $(MAKE) --no-print-directory '$m[$d,$t]' PROJECT=PhD \
	    START_DATE=19700101 END_DATE=20251231 ; )))
	@$(foreach a,$(ASSOCS),$(foreach t,$(THRESHOLDS), \
	  echo "Submitting NLL1D[$a,PhaseNet,INSTANCE,$t] to SLURM..."; \
	  $(MAKE) --no-print-directory 'NLL1D[$a,PhaseNet,INSTANCE,$t]' \
	    PROJECT=PhD \
	    START_DATE=20240320 END_DATE=20240620 ; ))

# =============================================================================
# CATALOG MANAGEMENT
# =============================================================================

# Merge manual OGS pick files into a unified catalog.
parse: # 3
	$(PYTHON_BIN) $(OGS_PATH)/src/ogsparser.py \
		-f $(DATASET_PATH)/* \
		--merge \
	  -o $(WORK_PATH)/catalogs/$(notdir $(patsubst %/,%,$(DATASET_PATH)))

test: # 2
	$(SBC_RUN_BIN) output_path=$(base)/$(YEAR)

# WARNING: deletes every generated catalog under $(WORK_PATH)/catalogs
clean: #
	@test -n "$(WORK_PATH)" && \
	test "$(WORK_PATH)" != "/" || { \
		echo "ERROR: refusing to clean unsafe WORK_PATH=$(WORK_PATH)" >&2; exit 1; \
	}
	@echo "Removing generated files under $(WORK_PATH)"
	@cd "$(WORK_PATH)" && \
	rm -rf -- config data src test && \
	rm -rf -- ACTIVATEME.sh download.sh dummy.sh init.sh ktanakah.sh \
		LAUNCHME.sh LEONARDO.yml Makefile
	#@echo "Removing generated catalogs below $(WORK_PATH)/catalogs"
	#@test -n "$(WORK_PATH)" && rm -rf -- "$(WORK_PATH)/catalogs"/*
