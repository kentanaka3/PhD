#!/usr/bin/env bash

# Markdown Validator
# ==================
#
# Validate Markdown structure, local links, and project script references.
# The validator reads source files only and reports findings to standard error.
#
# USAGE
#   bash md_val.sh [--root DIR|--file FILE]

set -euo pipefail
umask 077

readonly SCRIPT="${LLM_HANDLER_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly ROOT="$(cd -- "$WORKSPACE/.." && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

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

# Detect stale references to project shell scripts in one Markdown file.
validate_script_references() { # 16
	local -r file="$1"
	local -r scan_root="$2"
	local -r display_file="$3"
	local failures=0 reference

	while IFS= read -r reference; do
		[[ -e "$scan_root/$reference" ]] || {
			printf 'ERR %s (stale or missing script reference: %s)\n' \
				"$display_file" "$reference" >&2
			((failures += 1))
		}
	done < <(grep -oHE '(LLM/scripts|OGS/utils/Leonardo|OGS/utils/\.local)/[A-Za-z0-9_.-]+\.sh' \
		"$file" | cut -d: -f2- | LC_ALL=C sort -u || true)
	return "$failures"
}

# Validate Markdown files found below a workspace root.
validate_markdown() { # 16
	local -r scan_root="$1"
	local failures=0 file

	while IFS= read -r -d '' file; do
		if (( $(awk '/^```|^~~~/ { fenced = !fenced } END { print fenced + 0 }' "$file") != 0 )); then
			printf 'ERR %s (unclosed fenced code block)\n' "${file#$scan_root/}" >&2
			((failures += 1))
		fi
		validate_markdown_links "$file" "${file#$scan_root/}" ||
			failures=$((failures + $?))
		validate_script_references "$file" "$scan_root" "${file#$scan_root/}" ||
			failures=$((failures + $?))
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f -name '*.md' -print0 | LC_ALL=C sort -z)
	return "$failures"
}

# Validate one Markdown input file.
validate_single() { # 18
	local -r file="$1"
	local failures=0

	[[ -f "$file" ]] || fail 1 "file does not exist: $file"
	case "$file" in
		*.md) ;;
		*) fail 2 "single-file Markdown validation supports *.md: $file" ;;
	esac
	if (( $(awk '/^```|^~~~/ { fenced = !fenced } END { print fenced + 0 }' "$file") != 0 )); then
		printf 'ERR %s (unclosed fenced code block)\n' "$file" >&2
		((failures += 1))
	fi
	validate_markdown_links "$file" "$file" || failures=$((failures + $?))
	validate_script_references "$file" "$ROOT" "$file" || failures=$((failures + $?))
	(( failures == 0 )) || fail 1 "single-file validation failed with $failures issue(s)"
	printf 'Validation passed: %s\n' "$file"
}

# Parse Markdown validation options and dispatch checks.
main() { # 40
	local scan_root="$ROOT"
	local file_path=""

	require_command awk
	require_command find
	require_command grep
	require_command sort

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
				fail 2 "Markdown validation is run through: bash $SCRIPT_DIR/validate.sh [--root DIR|--file FILE]"
				;;
			*)
				fail 2 "unknown Markdown validator option: $1"
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
	validate_markdown "$normalized_root"
}

main "$@"
