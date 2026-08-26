# =============================================================================
# AISeism Seismic Data Processing Pipeline - Makefile
# =============================================================================
# Pipeline stages:
#   1. DATA ACQUISITION   : download, compress, decompress, retrieve, station
#   2. PHASE PICKING      : PhaseNet / EQTransformer with dataset+threshold args
#   3. PHASE ASSOCIATION  : PyOcto / GaMMA with picker-input args
#   4. EVENT LOCATION     : NLL1D / NLL3D with associator+picker-input args
#   5. CATALOG MANAGEMENT : parse, clean
#
# USAGE EXAMPLES:
#   make download YEAR=2025 STEP_DAYS=5 OFFSET=0 END_DAY=365
#   make download YEAR=2025 STEP_DAYS=5 OFFSET=0 END_DAY=365 THREADS=4
#	  make compress           # Compress raw waveforms into yearly tar.gz files
#	  make decompress         # Decompress yearly tar.gz files into raw waveforms
#   make index              # Build/update Squirrel indexes for the waveform archive
#   make PyOcto             # Run PyOcto with the default picker input
#   make GaMMA              # Run GaMMA with the default picker input
#   make REAL               # Run REAL with the default picker input
#   make NLL1D              # Run NonLinLoc 1D with the default assoc+picking inputs
#   make NLL3D              # Run NonLinLoc 3D with the default assoc+picking inputs
#   make base               # Just make the base catalog for quick iteration
#   make station YEAR=2025
#   make parse
#   make clean              # WARNING: deletes catalogs
#
# Database/Picker/associator/NLL targets are generated from the registered
# stage-name lists below via the
# _INDEXER_RULES / _PICKER_RULES / _ASSOC_RULES / _NLL_RULES macros.
#
# DEPENDENCIES:
#   - Analyze.sh         : SLURM job-submission wrapper (activates SBC env)
#   - ml_catalog_run     : ML catalog generation entrypoint (Hydra)
#   - $(SRC_PATH)/*.py   : OGS-specific Python scripts
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION (override at runtime: `make target VAR=value`)
# -----------------------------------------------------------------------------

# General
USER     ?= $(shell whoami)
ROOT     := $(CURDIR)
PYTHON   ?= python
# bash needed for the shell parameter expansion used in generated rules.
PROJECT  := ""
THREADS  ?= 4


# Data range
YEAR        ?= 2025
STEP_DAYS   ?= 5
OFFSET      ?= 0
END_DAY     ?= 366    # max 366 (leap years); loop clamps to year end

START_DATE ?= $(shell date -d "$(YEAR)-01-01 + $(OFFSET) days" +%Y%m%d)
END_DATE   ?= $(shell date -d "$(YEAR)-12-31" +%Y%m%d)

_START_DATE_CHECK := $(shell date -d "$(START_DATE)" +%Y%m%d 2>/dev/null || echo invalid)
_END_DATE_CHECK   := $(shell date -d "$(END_DATE)"   +%Y%m%d 2>/dev/null || echo invalid)
$(if $(filter invalid,$(_START_DATE_CHECK)),$(error Invalid START_DATE: $(START_DATE)))
$(if $(filter invalid,$(_END_DATE_CHECK)),$(error Invalid END_DATE: $(END_DATE)))

ifeq ($(YEAR),$(shell date -d "$(START_DATE)" +%Y))
else
	YEAR := $(shell date -d "$(START_DATE)" +%Y)
endif

# Paths
WORK_ROOT    ?= /leonardo_work/IscrC_AISeism/WORK
SRC_PATH     ?= /leonardo_work/IscrC_AISeism/PhD/OGS/src
WAVE_PATH    ?= $(WORK_ROOT)/waveform
MANUAL_PATH  ?= $(WORK_ROOT)/manual/OnlyEqNLL1D
CONDA_PATH   ?= /leonardo_work/IscrC_AISeism/.miniconda3/bin/conda
PYTHON_PATH  ?= /leonardo_work/IscrC_AISeism/.miniconda3/envs/SBC_3.12/bin/python
STATION_PATH ?= $(WAVE_PATH)/station
TT_DB        ?= $(ROOT)/OGS/data/VelocityModel/1D/tt_db.txt

# ml_catalog_run launcher — wrapped via Analyze.sh for SLURM submission.
# `$$@` in `define` bodies expands to the target name inside generated rules.
ML_CATALOG_RUN := HYDRA_FULL_ERROR=1 ml_catalog_run +libpath=.

# Default picker/associator roots that chained stages can reuse.
base        := OGS/$(YEAR)
BASE_PICKER := $(ROOT)/catalogs/$(base)/SeisBenchPicker
BASE_ASSOC  := $(ROOT)/catalogs/$(base)/PyOcto

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
.PHONY: help setup all base SeisBenchPicker parse test clean \
	download compress decompress station index \
	$(MODELS) $(ASSOCS) $(NLLS)

# -----------------------------------------------------------------------------
# DEFAULT / META
# -----------------------------------------------------------------------------
all: base
	@$(MAKE) --no-print-directory $(NLL)[$(ASSOC),$(MODEL),$(DATASET),$(THRESHOLD)] START_DATE=20240320 END_DATE=20240620

help:
	@echo "AISeism Makefile — common targets:"
	@echo "  make download   YEAR=YYYY STEP_DAYS=N OFFSET=D END_DAY=D"
	@echo "  make download   YEAR=YYYY STEP_DAYS=N OFFSET=D END_DAY=D THREADS=4"
	@echo "  make compress | decompress   YEAR=YYYY"
	@echo "  make station    YEAR=YYYY"
	@echo "  make index THREADS=8 START_DATE=1977-01-01 END_DATE=2025-12-31"
	@echo "  make PhaseNet | EQTransformer | ...	(picker models with default dataset/threshold)"
	@echo "  make PhaseNet[INSTANCE,0.1] | EQTransformer[STEAD,0.2] | ...	(picker models with dataset/threshold overrides)"
	@echo "  make PyOcto | GaMMA | REAL      			(associator with default picker input)"
	@echo "  make PyOcto[PhaseNet,INSTANCE,0.1] | GaMMA[EQTransformer,STEAD,0.2] | REAL[PhaseNet,SCEDC,0.3] ...	(associator with picker overrides)"
	@echo "  make NLL1D | NLL3D               	  (locator with default picker+assoc input)"
	@echo "  make NLL1D[PyOcto,PhaseNet,INSTANCE,0.1] | NLL3D[GaMMA,EQTransformer,STEAD,0.2] | ...	(locator with picker+assoc overrides)"
	@echo "  make base                     			 	(just make the base catalog for quick iteration on later stages)"
	@echo "  make parse | clean"

setup:
	@echo "Setup placeholder — module loads / env activation go here."

# =============================================================================
# DATA ACQUISITION
# =============================================================================

# Download waveforms in $(STEP_DAYS)-day chunks, clamped to $(YEAR).
download:
	@for i in $$(seq $(OFFSET) $(STEP_DAYS) $(END_DAY)); do \
		start=$$(date -d "$(YEAR)-01-01 + $$i days" +%Y%m%d); \
		end=$$(date -d "$(YEAR)-01-01 + $$((i+$(STEP_DAYS))) days" +%Y%m%d); \
		[ "$$(date -d "$(YEAR)-01-01 + $$i days" +%Y)" = "$(YEAR)" ] || continue; \
		[ "$$(date -d "$(YEAR)-01-01 + $$((i+$(STEP_DAYS))) days" +%Y)" = "$(YEAR)" ] || end=$(YEAR)1231; \
		echo "Downloading waveforms $$start → $$end (offset $$i days)"; \
		$(PYTHON_PATH) $(SRC_PATH)/ogsdownloader.py -D $$start $$end -d $(WAVE_PATH) \
			--threads $(THREADS) -S "* -SP -OL -ED"; \
	done

compress:
	$(PYTHON_PATH) $(WORK_ROOT)/compressor.py -C $(WAVE_PATH) . -v

decompress:
	$(PYTHON_PATH) $(WORK_ROOT)/compressor.py -D $(WAVE_PATH)/$(YEAR) \
		$(WAVE_PATH) -v && \
	find $(WAVE_PATH)/$(YEAR)/* -type f -name "*.tar.gz" -print -delete

# Station metadata for the configured date range.
station:
	@echo "Processing station metadata $(START_DATE) → $(END_DATE)..."
	$(PYTHON_PATH) $(SRC_PATH)/ogsstation.py \
		$(SRC_PATH) $(WAVE_PATH) $(STATION_PATH) \
		$(START_DATE) $(END_DATE) $(WORK_ROOT)/catalogs/$(base)

# Build/update the sharded Squirrel indexes for the full waveform archive.
index:
	$(ML_CATALOG_RUN) +index_only=true data=ogsDB data.check=true \
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
	$$(ML_CATALOG_RUN) +index_only=true data=ogsDB data.check=true \
		data.index_workers=$$(THREADS) \
		data.starttime=$$(START_DATE) data.endtime=$$(END_DATE)
endef

# --- PICKER ------------------------------------------------------------------
# id convention: <Model>[<Dataset>,<Threshold>]
define _PICKER_RULES
$1: ; @$$(MAKE) --no-print-directory '$1[$$(DATASET),$$(THRESHOLD)]'

$1[%]:
	@stem='$$*'; dataset="$$$${stem%%,*}"; threshold="$$$${stem##*,}"; \
	picker_id="$$$$dataset/$$$$threshold/$$(YEAR)"; \
	mkdir -p "$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$picker_id/SeisBenchPicker"; \
	$$(ML_CATALOG_RUN) output_path="catalogs/$$(PROJECT)/$1/$$$$picker_id" \
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
	picker_id="$$$$model/$$$$dataset/$$$$threshold/$$(YEAR)"; \
	mkdir -p "$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$picker_id"; \
	mkdir -p "$$(ROOT)/catalogs/$$(PROJECT)/$$$$picker_id"; \
	ln -sfn "$$(ROOT)/catalogs/$$(PROJECT)/$$$$picker_id/SeisBenchPicker" \
		"$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$picker_id/SeisBenchPicker"; \
	$$(ML_CATALOG_RUN) output_path="catalogs/$$(PROJECT)/$1/$$$$picker_id" \
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
	picker_id="$$$$model_/$$$$dataset_/$$$$threshold_/$$(YEAR)"; \
	assoc_id="$$$$assoc_/$$$$picker_id"; \
	mkdir -p "$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$assoc_id"; \
	mkdir -p "$$(ROOT)/catalogs/$$(PROJECT)/$$$$assoc_id/OGSPickStatQC"; \
	ln -sfn "$$(ROOT)/catalogs/$$(PROJECT)/$$$$assoc_id/OGSPickStatQC" \
		"$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$assoc_id/OGSPickStatQC"; \
	ln -sfn "$$(ROOT)/catalogs/$$(PROJECT)/$$$$picker_id/SeisBenchPicker" \
		"$$(ROOT)/catalogs/$$(PROJECT)/$1/$$$$assoc_id/SeisBenchPicker"; \
	$$(ML_CATALOG_RUN) output_path="catalogs/$$(PROJECT)/$1/$$$$assoc_id" \
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
	mkdir -p "$(ROOT)/catalogs/$(base)/$(YEAR)"

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

# =============================================================================
# CATALOG MANAGEMENT
# =============================================================================

# Merge manual OGS pick files into a unified catalog.
parse:
	$(PYTHON_PATH) $(SRC_PATH)/ogsparser.py -f $(MANUAL_PATH)/* --merge \
		-o $(WORK_ROOT)/catalogs/$(notdir $(patsubst %/,%,$(MANUAL_PATH)))

test:
	$(ML_CATALOG_RUN) output_path=$(base)/$(YEAR)

# WARNING: deletes every generated catalog under $(ROOT)/catalogs
clean:
	rm -rf $(ROOT)/catalogs/*/assignments/*
	rm -rf $(ROOT)/catalogs/*/events/*
	rm -rf $(ROOT)/catalogs/*
