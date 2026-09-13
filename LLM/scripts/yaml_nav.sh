#!/usr/bin/env bash

# YAML Navigator
# ==============
#
# Navigate YAML-named documents by listing conservative block-mapping paths or
# extracting structural blocks and explicit line ranges. This is not a YAML
# validator or full parser: it recognizes only plain block mapping keys, skips
# comments, and ignores block-scalar contents. Source input files are read only.
#
# USAGE
#   bash yaml_nav.sh outline FILE [--depth MAX_DEPTH] [--from START_LINE] [--to END_LINE]
#   bash yaml_nav.sh get FILE START_LINE
#   bash yaml_nav.sh slice FILE START_LINE END_LINE
#   bash yaml_nav.sh --help
#
# DESCRIPTION
#   `outline` writes recognized paths, absolute line numbers, and structural
#   depths to standard output. `get` writes a recognized mapping entry and its
#   indented block through the next peer or parent entry. `slice` writes an
#   inclusive source line range. Diagnostics and usage errors are written to
#   standard error by the shared helpers. Every command is safe to repeat.
#
# COMMANDS
#   outline   Print conservative plain-key block-mapping paths.
#   get       Extract the block rooted at a recognized mapping line.
#   slice     Extract an explicit inclusive source line range.
#
# Function                    | description
# ----------------------------|---------------------------------------------
# usage                       | Print command-line usage information.
# validate_yaml_file          | Verify a readable YAML-named input file.
# validate_positive_integer   | Verify a positive decimal integer argument.
# validate_navigation_config  | Check command-specific executable prerequisites.
# run_outline                 | Print recognized mapping paths outside block scalars.
# run_get                     | Print the block rooted at a mapping line.
# run_slice                   | Print an inclusive source line range.
# run_navigation_command      | Parse commands, validate configuration, and dispatch.
# main                        | Delegate command handling to the common navigator.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Print command-line usage information.
usage() { # 22
	cat <<EOF
Usage: $SCRIPT COMMAND [OPTIONS]

Commands:
  outline FILE [--depth MAX_DEPTH] [--from START_LINE] [--to END_LINE]
	  Print conservative plain-key paths with absolute line numbers.
  get FILE START_LINE
	  Print a recognized mapping entry through the next peer or parent entry.
  slice FILE START_LINE END_LINE
	  Print the inclusive source line range.

Options:
  -h, --help    Show this help message and exit.
  --            End option processing.

Examples:
  bash $SCRIPT outline config/example.yaml --depth 2
  bash $SCRIPT get config/example.yaml 5
  bash $SCRIPT slice config/example.yaml 1 20
EOF
}

# Verify that a readable YAML-named regular file exists.
validate_yaml_file() { # 10
	local -r file="$1"

	[[ -n "$file" ]] || fail 2 "missing YAML file"
	[[ -f "$file" && -r "$file" ]] || fail 1 "file is not readable: $file"
	case "$file" in
		*.yaml|*.yml) ;;
		*) fail 2 "file must use a .yaml or .yml extension: $file" ;;
	esac
}

# Print recognized mapping paths outside block scalars.
run_outline() { # 109
	local -r file="${1:-}"
	local max_depth=99
	local from_line=1
	local to_line=999999999

	[[ $# -ge 1 ]] || fail 2 "outline requires FILE"
	shift
	validate_yaml_file "$file"

	while [[ $# -gt 0 ]]; do
		case "$1" in
			-h|--help)
				usage
				return 0
				;;
			--depth)
				[[ $# -ge 2 ]] || fail 2 "--depth requires MAX_DEPTH"
				max_depth="$2"
				shift 2
				;;
			--from)
				[[ $# -ge 2 ]] || fail 2 "--from requires START_LINE"
				from_line="$2"
				shift 2
				;;
			--to)
				[[ $# -ge 2 ]] || fail 2 "--to requires END_LINE"
				to_line="$2"
				shift 2
				;;
			--)
				shift
				[[ $# -eq 0 ]] || fail 2 "unexpected argument: $1"
				;;
			*)
				fail 2 "unknown outline option: $1"
				;;
		esac
	done

	validate_positive_integer "$max_depth" "MAX_DEPTH"
	validate_positive_integer "$from_line" "START_LINE"
	validate_positive_integer "$to_line" "END_LINE"
	(( from_line <= to_line )) || fail 2 "START_LINE must not exceed END_LINE"

	awk -v max_depth="$max_depth" -v from_line="$from_line" -v to_line="$to_line" '
		function strip_comment(text,    position, character, previous, single_quote, double_quote, escaped) {
			for (position = 1; position <= length(text); position++) {
				character = substr(text, position, 1)
				previous = position == 1 ? "" : substr(text, position - 1, 1)
				if (character == "\"" && !single_quote && !escaped) double_quote = !double_quote
				if (character == "\047" && !double_quote) single_quote = !single_quote
				if (character == "#" && !single_quote && !double_quote && (position == 1 || previous ~ /[[:space:]]/)) return substr(text, 1, position - 1)
				escaped = character == "\\" && !escaped
				if (character != "\\") escaped = 0
			}
			return text
		}
		function reset_stack(    position) {
			for (position = 1; position <= max_seen_depth; position++) {
				delete path_part[position]
				delete path_indent[position]
			}
			max_seen_depth = 0
		}
		{
			text = strip_comment($0)
			match(text, /^[[:space:]]*/)
			indentation = RLENGTH
			content = substr(text, indentation + 1)
			if (content ~ /^(---|\.\.\.)[[:space:]]*$/) {
				reset_stack()
				next
			}
			if (in_block_scalar) {
				if (content != "" && indentation <= block_indent) in_block_scalar = 0
				else next
			}
			sequence_width = 0
			if (match(content, /^-[[:space:]]+/)) {
				sequence_width = RLENGTH
				content = substr(content, RLENGTH + 1)
			}
			if (content !~ /^[A-Za-z_][A-Za-z0-9_.-]*[[:space:]]*:/) next
			key = content
			sub(/[[:space:]]*:.*/, "", key)
			key_indent = indentation + sequence_width
			depth = 1
			while (depth <= max_seen_depth && path_indent[depth] < key_indent) depth++
			for (position = depth; position <= max_seen_depth; position++) {
					delete path_part[position]
					delete path_indent[position]
			}
			path_part[depth] = key
			path_indent[depth] = key_indent
			max_seen_depth = depth
			path = path_part[1]
			for (position = 2; position <= depth; position++) path = path "." path_part[position]
			if (NR >= from_line && NR <= to_line && depth <= max_depth) {
				printf "%-6d | D%d | %s\n", NR, depth, path
			}
			if (content ~ /^[A-Za-z_][A-Za-z0-9_.-]*[[:space:]]*:[[:space:]]*[>|][+-]?[0-9]*[[:space:]]*$/) {
				in_block_scalar = 1
				block_indent = indentation
			}
		}
	' "$file"
}

# Print the block rooted at a recognized YAML mapping line.
run_get() { # 60
	local -r file="${1:-}"
	local -r target_line="${2:-}"

	[[ $# -eq 2 ]] || fail 2 "get requires FILE and START_LINE"
	validate_yaml_file "$file"
	validate_positive_integer "$target_line" "START_LINE"

	if ! awk -v target_line="$target_line" '
		function strip_comment(text,    position, character, previous, single_quote, double_quote, escaped) {
			for (position = 1; position <= length(text); position++) {
				character = substr(text, position, 1)
				previous = position == 1 ? "" : substr(text, position - 1, 1)
				if (character == "\"" && !single_quote && !escaped) double_quote = !double_quote
				if (character == "\047" && !double_quote) single_quote = !single_quote
				if (character == "#" && !single_quote && !double_quote && (position == 1 || previous ~ /[[:space:]]/)) return substr(text, 1, position - 1)
				escaped = character == "\\" && !escaped
				if (character != "\\") escaped = 0
			}
			return text
		}
		{
			text = strip_comment($0)
			match(text, /^[[:space:]]*/)
			indentation = RLENGTH
			content = substr(text, indentation + 1)
			if (capturing && content ~ /^(---|\.\.\.)[[:space:]]*$/) exit
			if (in_block_scalar) {
				if (content != "" && indentation <= block_indent) in_block_scalar = 0
				else {
					if (capturing) print
					next
				}
			}
			sequence_width = 0
			if (match(content, /^-[[:space:]]+/)) {
				sequence_width = RLENGTH
				content = substr(content, RLENGTH + 1)
			}
			is_mapping = content ~ /^[A-Za-z_][A-Za-z0-9_.-]*[[:space:]]*:/
			if (NR == target_line) {
				if (!is_mapping) exit 3
				capturing = 1
				target_indent = indentation
				found = 1
				print
			} else if (capturing) {
				if (is_mapping && indentation <= target_indent) exit
				print
			}
			if (is_mapping && content ~ /^[A-Za-z_][A-Za-z0-9_.-]*[[:space:]]*:[[:space:]]*[>|][+-]?[0-9]*[[:space:]]*$/) {
				in_block_scalar = 1
				block_indent = indentation
			}
		}
		END { if (!found) exit 3 }
	' "$file"; then
		fail 2 "START_LINE does not identify a recognized YAML mapping: $target_line"
	fi
}

# Print an inclusive source line range.
run_slice() { # 12
	local -r file="${1:-}"
	local -r start_line="${2:-}"
	local -r end_line="${3:-}"

	[[ $# -eq 3 ]] || fail 2 "slice requires FILE, START_LINE, and END_LINE"
	validate_yaml_file "$file"
	validate_positive_integer "$start_line" "START_LINE"
	validate_positive_integer "$end_line" "END_LINE"
	(( start_line <= end_line )) || fail 2 "START_LINE must not exceed END_LINE"
	sed -n "${start_line},${end_line}p" -- "$file"
}

# Delegate command handling to the common navigator.
main() { # 3
	run_navigation_command usage run_outline run_get run_slice "$@"
}

main "$@"
