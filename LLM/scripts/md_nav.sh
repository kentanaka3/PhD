#!/usr/bin/env bash

# Markdown Navigator
# ==================
#
# Navigate Markdown documents by listing headers or extracting sections and
# explicit line ranges. The script treats fenced code blocks as content, not
# structure, so example headings cannot affect outline or section boundaries.
#
# USAGE
#   bash md_nav.sh outline FILE [--depth MAX_DEPTH] [--from START_LINE] [--to END_LINE]
#   bash md_nav.sh get FILE START_LINE
#   bash md_nav.sh slice FILE START_LINE END_LINE
#   bash md_nav.sh --help
#
# DESCRIPTION
#   `outline` writes headings, their absolute line numbers, and depths to
#   standard output. `get` writes a header and its content through the next
#   heading of equal or higher prominence. `slice` writes an inclusive line
#   range. The source Markdown file is read only, making every command safe to
#   repeat. Line numbers and depths must be positive decimal integers.
#
# COMMANDS
#   outline   Print code-fence-safe headings with absolute lines and depths.
#   get       Extract a section beginning at the specified heading line.
#   slice     Extract an explicit inclusive line range.
#
# Function                   | description
# ---------------------------|----------------------------------------------
# log                        | Print a timestamped diagnostic message.
# fail                       | Report an error and terminate.
# require_command            | Verify that a required executable is available.
# usage                      | Print command-line usage information.
# validate_file              | Verify that a readable Markdown input exists.
# validate_positive_integer  | Verify a positive decimal integer argument.
# validate_navigation_config | Check command-specific executable prerequisites.
# run_outline                | Print eligible headings outside fenced code blocks.
# run_get                    | Print the section rooted at a heading line.
# run_slice                  | Print an inclusive source line range.
# run_navigation_command     | Parse commands, validate configuration, and dispatch.
# main                       | Delegate command handling to the common navigator.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# usage
# -----
# Print command-line usage information.
usage() { # 22
	cat <<EOF
Usage: $SCRIPT COMMAND [OPTIONS]

Commands:
  outline FILE [--depth MAX_DEPTH] [--from START_LINE] [--to END_LINE]
	  Print headings outside fenced code blocks with absolute line numbers.
  get FILE START_LINE
	  Print a heading and its section through the next peer or parent heading.
  slice FILE START_LINE END_LINE
	  Print the inclusive source line range.

Options:
  -h, --help    Show this help message and exit.
  --            End option processing.

Examples:
  bash $SCRIPT outline LLM/README.md --depth 2
  bash $SCRIPT get LLM/README.md 5
  bash $SCRIPT slice LLM/README.md 1 20
EOF
}

# validate_file
# -------------
# Verify that a readable regular file exists.
validate_file() { # 6
	local -r file="$1"

	[[ -n "$file" ]] || fail 2 "missing Markdown file"
	[[ -f "$file" && -r "$file" ]] || fail 1 "file is not readable: $file"
}

# Print eligible headings outside fenced code blocks.
run_outline() { # 64
	local -r file="${1:-}"
	local max_depth=6
	local from_line=1
	local to_line=999999999

	[[ $# -ge 1 ]] || fail 2 "outline requires FILE"
	shift
	validate_file "$file"

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
	(( max_depth <= 6 )) || fail 2 "MAX_DEPTH must not exceed 6: $max_depth"
	(( from_line <= to_line )) || fail 2 "START_LINE must not exceed END_LINE"

	awk -v max_depth="$max_depth" -v from_line="$from_line" -v to_line="$to_line" '
		BEGIN { in_code = 0 }
		/^[ \t]*(`{3,}|~{3,})/ {
			in_code = !in_code
			next
		}
		!in_code && NR >= from_line && NR <= to_line && /^[ \t]*#{1,6}[ \t]+/ {
			match($0, /#{1,6}/)
			depth = RLENGTH
			if (depth <= max_depth) {
				title = $0
				sub(/^[ \t]*#{1,6}[ \t]+/, "", title)
				printf "%-6d | L%d | %s\n", NR, depth, title
			}
		}
	' "$file"
}

# Print the section rooted at a Markdown heading line.
run_get() { # 31
	local -r file="${1:-}"
	local -r target_line="${2:-}"

	[[ $# -eq 2 ]] || fail 2 "get requires FILE and START_LINE"
	validate_file "$file"
	validate_positive_integer "$target_line" "START_LINE"

	awk -v target_line="$target_line" '
		BEGIN { in_code = 0; target_depth = 0; capturing = 0 }
		/^[ \t]*(`{3,}|~{3,})/ {
			in_code = !in_code
			if (capturing) print
			next
		}
		!in_code && /^[ \t]*#{1,6}[ \t]+/ {
			match($0, /#{1,6}/)
			current_depth = RLENGTH
			if (NR == target_line) {
				target_depth = current_depth
				capturing = 1
				print
				next
			}
			if (capturing && current_depth <= target_depth) {
				exit
			}
		}
		capturing { print }
	' "$file"
}

# Print an inclusive source line range.
run_slice() { # 12
	local -r file="${1:-}"
	local -r start_line="${2:-}"
	local -r end_line="${3:-}"

	[[ $# -eq 3 ]] || fail 2 "slice requires FILE, START_LINE, and END_LINE"
	validate_file "$file"
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
