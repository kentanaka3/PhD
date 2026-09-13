#!/usr/bin/env bash

# YAML Validator
# ==============
#
# Parse YAML documents with Ruby's standard Psych parser. The validator reads
# source files only and reports syntax findings to standard error.
#
# USAGE
#   bash yaml_val.sh [--root DIR|--file FILE]

set -euo pipefail
umask 077

readonly SCRIPT="${LLM_HANDLER_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly ROOT="$(cd -- "$WORKSPACE/.." && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Parse one YAML file without constructing application objects.
validate_yaml_file() { # 14
	local -r file="$1"
	local -r display_file="$2"

	[[ -f "$file" ]] || fail 1 "file does not exist: $file"
	case "$file" in
		*.yaml|*.yml) ;;
		*) fail 2 "single-file YAML validation supports *.yaml and *.yml: $file" ;;
	esac
	if ! ruby -ryaml -e 'YAML.parse_stream(File.read(ARGV.fetch(0)))' -- "$file"; then
		printf 'ERR %s (YAML parse check failed)\n' "$display_file" >&2
		return 1
	fi
}

# Validate YAML files found below a workspace root.
validate_yaml() { # 11
	local -r scan_root="$1"
	local failures=0 file

	while IFS= read -r -d '' file; do
		validate_yaml_file "$file" "${file#$scan_root/}" ||
			failures=$((failures + $?))
	done < <(find "$scan_root" -path "$scan_root/.git" -prune -o -type f \
		\( -name '*.yaml' -o -name '*.yml' \) -print0 | LC_ALL=C sort -z)
	return "$failures"
}

# Parse YAML validation options and dispatch checks.
main() { # 40
	local scan_root="$ROOT"
	local file_path=""

	require_command find
	require_command ruby
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
				fail 2 "YAML validation is run through: bash $SCRIPT_DIR/validate.sh [--root DIR|--file FILE]"
				;;
			*)
				fail 2 "unknown YAML validator option: $1"
				;;
		esac
	done

	if [[ -n "$file_path" ]]; then
		validate_yaml_file "$file_path" "$file_path"
		printf 'Validation passed: %s\n' "$file_path"
		return 0
	fi

	local normalized_root
	normalized_root="$(cd -- "$scan_root" 2>/dev/null && pwd -P)" ||
		fail 1 "could not resolve validation root: $scan_root"
	validate_yaml "$normalized_root"
}

main "$@"
