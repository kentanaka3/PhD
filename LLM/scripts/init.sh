#!/usr/bin/env bash

# LLM Workspace Initializer
# =========================
#
# Create the missing LLM workspace directories and starter Markdown records.
# Existing files are preserved. Use --dry-run to report intended mutations.
#
# USAGE
#   bash init.sh [--dry-run]
#
# OPTIONS
#   --dry-run    Report files and directories that would be created.
#
# Function     | description
# -------------|------------------------------------------------------------
# initialize   | Create missing workspace directories and starter records.
# main         | Parse arguments, validate prerequisites, and initialize.

set -euo pipefail
umask 077

readonly SCRIPT="${LLM_HANDLER_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# initialize
# ----------
# Create missing workspace directories and starter records.
initialize() { # 42
	local -r dry_run="$1"
	local -a directories=(
		00_governance 01_context 02_prompts 03_workflows 04_experiments
		05_evaluations 06_outputs 07_archive config scripts templates
	)
	local -A starter_files=(
		["04_experiments/README.md"]=$'# Experiment records\n\nUse ../templates/experiment.md for each meaningful run.\n'
		["06_outputs/README.md"]=$'# Reviewed outputs\n\nRecord reviewer, date, sources, and allowed use.\n'
		["07_archive/README.md"]=$'# Archive\n\nMove superseded material here; preserve research history.\n'
	)
	local directory relative_path target

	require_command mkdir
	for directory in "${directories[@]}"; do
		target="$WORKSPACE/$directory"
		if [[ -d "$target" ]]; then
			[[ "$dry_run" == true ]] && log "Preserved directory: ${target#$WORKSPACE/}"
		elif [[ -e "$target" || -L "$target" ]]; then
			fail 1 "path exists and is not a directory: $target"
		elif [[ "$dry_run" == true ]]; then
			log "Would create directory: $target"
		else
			mkdir -p -- "$target"
		fi
	done

	for relative_path in "${!starter_files[@]}"; do
		target="$WORKSPACE/$relative_path"
		if [[ ! -e "$target" && "$dry_run" == false ]]; then
			printf '%s' "${starter_files[$relative_path]}" > "$target"
			log "Created ${target#$WORKSPACE/}"
		elif [[ ! -e "$target" ]]; then
			log "Would create file: ${target#$WORKSPACE/}"
		else
			log "Preserved ${target#$WORKSPACE/}"
		fi
	done

	printf '\nLLM workspace ready: %s\nNext: review 00_governance/open_questions.md and run the validator.\n' \
		"$WORKSPACE"
}

# main
# ----
# Parse arguments, validate prerequisites, and initialize the workspace.
main() { # 14
	local dry_run=false

	require_command date
	require_command printf

	if [[ "${1:-}" == "--dry-run" ]]; then
		dry_run=true
		shift
	fi
	(( $# == 0 )) || fail 2 "unknown init option: $1"

	initialize "$dry_run"
}

main "$@"
