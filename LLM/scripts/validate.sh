#!/usr/bin/env bash

# =============================================================================
# LLM Workspace Validator
# =============================================================================
#
# Validate the required LLM documentation scaffold and managed Bash scripts.
# The validator reads source files only and reports findings to standard error.
# Markdown and YAML checks are delegated to md_val.sh and yaml_val.sh.
#
# USAGE
#   bash validate.sh [--root DIR|--file FILE]
#
# OPTIONS
#   --root DIRECTORY    Validate the complete workspace rooted at DIRECTORY.
#   --file FILE         Validate one Bash script, Markdown document, or YAML file.
#
# Function                   | description
# ---------------------------|----------------------------------------------
# validate_markdown_links    | Validate local Markdown paths in one file.
# validate_markdown          | Validate local Markdown paths and code fences.
# validate_scripts           | Detect stale shell-script references.
# validate_bash              | Check Bash syntax under a root.
# validate_functions         | Check Bash function documentation and counts.
# validate_single            | Validate one supported Bash or Markdown file.
# validate_managed_functions | Check function counts in every managed script.
# validate                   | Validate the required workspace and scripts.
# main                       | Parse validation options and dispatch checks.
#
# AUTHORS:
#   - 健
#   - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
#     Centro di Ricerche Sismologiche (CRS)
#   - Università degli Studi di Trieste (UniTS)
#     Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
#     Applied Data Science and Artificial Intelligence (ADSAI)
#   - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
#     Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
#
# =============================================================================

set -euo pipefail
umask 077

readonly SCRIPT="${LLM_HANDLER_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly ROOT="$(cd -- "$WORKSPACE/.." && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# validate_markdown_links
# -----------------------
# Validate local Markdown paths in one file.
validate_markdown_links() { # 26
	local -r file="$1"
	local -r display_file="$2"
	local failures=0 line_number link target

	while IFS=$'\t' read -r line_number link; do
		[[ -n "$link" && "$link" != http* && "$link" != mailto:* ]] || continue
		target="$(dirname -- "$file")/$link"
		[[ -e "$target" ]] || {
			printf 'ERR %s:%s (broken Markdown path: %s)\n' \
				"$display_file" "$line_number" "$link" >&2
			((failures += 1))
		}
	done < <(awk '
		{
			remaining = $0
			while (match(remaining, /\]\([^)#]+(#[^)]*)?\)/)) {
				link = substr(remaining, RSTART + 2, RLENGTH - 3)
				sub(/#.*/, "", link)
				printf "%d\t%s\n", FNR, link
				remaining = substr(remaining, RSTART + RLENGTH)
			}
		}
	' "$file")
	return "$failures"
}

# validate_markdown
# -----------------
# Validate local Markdown paths and fenced code blocks.
validate_markdown() { # 14
	local -r scan_root="$1"
	local failures=0 file

	while IFS= read -r -d '' file; do
		if (( $(awk '/^```|^~~~/ { fenced = !fenced } END { print fenced + 0 }' "$file") != 0 )); then
			printf 'ERR %s (unclosed fenced code block)\n' "${file#$scan_root/}" >&2
			((failures += 1))
		fi
		validate_markdown_links "$file" "${file#$scan_root/}" ||
			failures=$((failures + $?))
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f -name '*.md' -print0 | LC_ALL=C sort -z)
	return "$failures"
}

# validate_scripts
# ----------------
# Detect stale references to project shell scripts.
validate_scripts() { # 16
	local -r scan_root="$1"
	local failures=0 file reference

	while IFS= read -r -d '' file; do
		while IFS= read -r reference; do
			[[ -e "$scan_root/$reference" ]] || {
				printf 'ERR %s (stale or missing script reference: %s)\n' \
					"${file#$scan_root/}" "$reference" >&2
				((failures += 1))
			}
		done < <(grep -oHE '(LLM/scripts|OGS/utils/Leonardo|OGS/utils/\.local)/[A-Za-z0-9_.-]+\.sh' \
			"$file" | cut -d: -f2- | LC_ALL=C sort -u || true)
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f -name '*.md' -print0 | LC_ALL=C sort -z)
	return "$failures"
}

# validate_bash
# -------------
# Check the syntax of every Bash script under a root.
validate_bash() { # 13
	local -r scan_root="$1"
	local failures=0 file

	require_command bash
	while IFS= read -r -d '' file; do
		if ! bash -n "$file"; then
			printf 'ERR %s (Bash syntax check failed)\n' "${file#$scan_root/}" >&2
			((failures += 1))
		fi
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f -name '*.sh' -print0 | LC_ALL=C sort -z)
	return "$failures"
}

# validate_functions
# ------------------
# Check inline Bash function documentation line counts in one file.
validate_functions() { # 24
	local -r target_file="$1"
	local -r scan_root="$(dirname -- "$target_file")"
	local failures=0 file line declaration count actual

	while IFS= read -r -d '' file; do
		while IFS=: read -r line declaration; do
			count=$(printf '%s\n' "$declaration" | sed -nE 's/.*\{[[:space:]]*#[[:space:]]*([0-9]+).*/\1/p')
			if [[ -z "$count" ]]; then
				printf 'ERR %s:%s (function declaration must end with: { # line_count)\n' \
					"${file#$scan_root/}" "$line" >&2
				((failures += 1))
				continue
			fi
			actual=$(awk -v start="$line" 'NR > start && /^}$/ { print NR - start + 1; exit }' "$file")
			if [[ "$actual" != "$count" ]]; then
				printf 'ERR %s:%s (documented %s lines, counted %s)\n' \
					"${file#$scan_root/}" "$line" "$count" "${actual:-unknown}" >&2
				((failures += 1))
			fi
		done < <(grep -nE '^[[:alnum:]_]+\(\)[[:space:]]*\{' "$file" || true)
	done < <(printf '%s\0' "$target_file")
	return "$failures"
}

# validate_single
# ---------------
# Validate one supported Bash script or Markdown file.
validate_single() { # 35
	local -r file="$1"
	local failures=0 reference

	[[ -f "$file" ]] || fail 1 "file does not exist: $file"
	case "$file" in
		*.sh)
			if ! bash -n "$file"; then
				printf 'ERR %s (Bash syntax check failed)\n' "$file" >&2
				((failures += 1))
			fi
			validate_functions "$file" || failures=$((failures + $?))
			;;
		*.md)
			if (( $(awk '/^```|^~~~/ { fenced = !fenced } END { print fenced + 0 }' "$file") != 0 )); then
				printf 'ERR %s (unclosed fenced code block)\n' "$file" >&2
				((failures += 1))
			fi
			validate_markdown_links "$file" "$file" ||
				failures=$((failures + $?))
			while IFS= read -r reference; do
				[[ -e "$ROOT/$reference" ]] || {
					printf 'ERR %s (stale or missing script reference: %s)\n' "$file" "$reference" >&2
					((failures += 1))
				}
			done < <(grep -oHE '(LLM/scripts|OGS/utils/Leonardo|OGS/utils/\.local)/[A-Za-z0-9_.-]+\.sh' "$file" | cut -d: -f2- | LC_ALL=C sort -u || true)
			;;
		*)
			fail 2 "single-file validation supports Bash scripts (*.sh) and Markdown (*.md): $file"
			;;
	esac

	(( failures == 0 )) || fail 1 "single-file validation failed with $failures issue(s)"
	printf 'Validation passed: %s\n' "$file"
}

# validate_managed_functions
# --------------------------
# Check function counts in every managed LLM command script.
validate_managed_functions() { # 13
	local -r scan_root="$1"
	local failures=0 script
	local -a managed_scripts=(
		common.sh handler.sh init.sh navigate.sh validate.sh
	)

	for script in "${managed_scripts[@]}"; do
		validate_functions "$scan_root/LLM/scripts/$script" ||
			failures=$((failures + $?))
	done
	return "$failures"
}

# validate
# --------
# Validate repository Markdown and Bash files plus critical entry points.
validate() { # 36
	local -r scan_root="$1"
	local failures=0 file files_checked=0

	for file in "$scan_root/AGENTS.md" "$scan_root/LLM/README.md" \
		"$scan_root/LLM/scripts/handler.sh"; do
		[[ -s "$file" ]] && continue
		printf 'ERR %s (missing or empty critical entry point)\n' \
			"${file#$scan_root/}" >&2
		((failures += 1))
	done

	while IFS= read -r -d '' file; do
		if [[ -s "$file" ]]; then
			printf 'OK  %s\n' "${file#$scan_root/}"
		else
			printf 'ERR %s (missing or empty)\n' "${file#$scan_root/}" >&2
			((failures += 1))
		fi
		((files_checked += 1))
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f \
		\( -name '*.md' -o -name '*.sh' \) -print0 | LC_ALL=C sort -z)

	[[ -x "$scan_root/LLM/scripts/handler.sh" ]] || {
		printf 'ERR LLM/scripts/handler.sh is not executable\n' >&2
		((failures += 1))
	}

	validate_markdown "$scan_root" || failures=$((failures + $?))
	validate_scripts "$scan_root" || failures=$((failures + $?))
	validate_bash "$scan_root" || failures=$((failures + $?))
	validate_managed_functions "$scan_root" || failures=$((failures + $?))

	(( failures == 0 )) || fail 1 "validation failed with $failures issue(s)"
	printf '\nValidation passed: %d Markdown and Bash files checked.\n' "$files_checked"
}

# Parse validation options and dispatch the requested check.
main() { # 38
	local scan_root="$ROOT"
	local file_path=""

	require_command date
	require_command printf

	while [[ $# -gt 0 ]]; do
		case "$1" in
			--root)
				(( $# >= 2 )) || fail 2 "--root requires a directory argument"
				scan_root="$2"
				shift 2
				;;
			--file)
				(( $# >= 2 )) || fail 2 "--file requires a file argument"
				file_path="$2"
				shift 2
				;;
			-h|--help)
				fail 2 "validation help is available through: bash $SCRIPT_DIR/handler.sh --help"
				;;
			*)
				fail 2 "unknown validate option: $1"
				;;
		esac
	done

	if [[ -n "$file_path" ]]; then
		validate_single "$file_path"
		return 0
	fi

	local normalized_root
	normalized_root="$(cd -- "$scan_root" 2>/dev/null && pwd -P)" ||
		fail 1 "could not resolve validation root: $scan_root"
	validate "$normalized_root"
}

main "$@"
