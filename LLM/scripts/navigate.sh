#!/usr/bin/env bash

# LLM Context Manifest Navigator
# ==============================
#
# Generate deterministic Tier 1 repository routing or Tier 2 scoped manifests.
# Navigation is read-only and writes YAML only to standard output.
#
# USAGE
#   bash navigate.sh [--root DIR] [--module PATH] [--include-assets] [--scripts]
#
# OPTIONS
#   --root DIRECTORY      Scan DIRECTORY instead of the repository root.
#   --module REL_PATH     Emit a Tier 2 manifest for a path relative to root.
#   --include-assets      Include supported image and PDF assets in Tier 2.
#   --scripts             Include a symbol index in Tier 2 mode only.
#
# Function           | description
# -------------------|-------------------------------------------------------
# yaml_escape        | Escape strings for safe YAML scalar emission.
# discover_validate  | Statically discover Makefile validation targets.
# scan_conflicts     | Detect repository or scoped-module structural issues.
# emit_tier1         | Emit the global routing manifest.
# extract_symbols    | Extract Make, Bash, and Python symbols in a module.
# emit_tier2         | Emit a scoped module manifest.
# main               | Parse navigation options and emit the requested manifest.

set -euo pipefail
umask 077

readonly SCRIPT="${LLM_HANDLER_SCRIPT_NAME:-$(basename -- "${BASH_SOURCE[0]}")}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly ROOT="$(cd -- "$WORKSPACE/.." && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# yaml_escape
# -----------
# Escape a string for safe YAML scalar emission.
yaml_escape() { # 10
	local str="$1"
	if [[ "$str" =~ [:\,\{\}\[\]\"\'\#\*\&\!\|\>\%\@\`] || "$str" =~ ^[[:space:]] || "$str" =~ [[:space:]]$ || -z "$str" ]]; then
		str="${str//\\/\\\\}"
		str="${str//\"/\\\"}"
		printf '"%s"' "$str"
	else
		printf '%s' "$str"
	fi
}

# discover_validate
# -----------------
# Statically discover validation targets from Makefiles in a directory.
discover_validate() { # 20
	local -r search_dir="$1"
	local makefile="$search_dir/Makefile"
	local -a targets=()

	if [[ -f "$makefile" ]]; then
		while IFS= read -r target; do
			[[ -n "$target" ]] && targets+=("$target")
		done < <(grep -E '^[A-Za-z0-9_.%+-]+[[:space:]]*:' "$makefile" | \
				 cut -d: -f1 | \
				 grep -E 'test|check|lint|validate|dry-run' | \
				 LC_ALL=C sort -u || true)
	fi

	if (( ${#targets[@]} == 0 )); then
		printf '[handler-validate]'
	else
		printf '[%s]' "$(IFS=,; echo "${targets[*]}")"
	fi
}

# scan_conflicts
# --------------
# Detect repository or scoped-module structural issues.
scan_conflicts() { # 22
	local -r scan_root="$1"
	local -r rel_module="${2:-}"
	local -a conflict_list=()
	local entry_file

	if [[ -z "$rel_module" ]]; then
		for entry_file in "OGS/README.md" "doc/README.md" "LLM/README.md"; do
			if [[ ! -f "$scan_root/$entry_file" ]]; then
				conflict_list+=("{type: missing_entry_point, target: \"$entry_file\"}")
			fi
		done
	elif [[ ! -d "$scan_root/$rel_module" && ! -f "$scan_root/$rel_module" ]]; then
		conflict_list+=("{type: missing_target, path: \"$rel_module\"}")
	fi

	if (( ${#conflict_list[@]} == 0 )); then
		printf '[]'
	else
		printf '[%s]' "$(IFS=,; echo "${conflict_list[*]}")"
	fi
}

# emit_tier1
# ----------
# Emit the Tier 1 global routing manifest.
emit_tier1() { # 22
	local -r scan_root="$1"
	local ogs_val doc_val llm_val conflicts

	ogs_val="$(discover_validate "$scan_root/OGS")"
	doc_val="$(discover_validate "$scan_root/doc")"
	llm_val="$(discover_validate "$scan_root/LLM")"
	conflicts="$(scan_conflicts "$scan_root")"

	cat <<EOF
schema: context-manifest/v2
root: .
tier: 1
authority: {executable: primary, conflicts: human_review}
modules:
  - {path: OGS, role: aggregate, entry: [OGS/README.md], validate: $ogs_val, ops: reviewed}
  - {path: doc, role: aggregate, entry: [doc/README.md], validate: $doc_val, ops: reviewed}
  - {path: LLM, role: governance, entry: [LLM/README.md], validate: $llm_val, ops: reviewed}
conflicts: $conflicts
EOF
	emit_navigation "$scan_root"
}

# extract_symbols
# ---------------
# Extract Make, Bash, and Python symbols from a scoped path.
extract_symbols() { # 30
	local -r base_dir="$1"
	local file line symbol

	printf 'symbols:\n'
	printf '  make_targets:\n'
	while IFS= read -r -d '' file; do
		while IFS=: read -r line symbol; do
			printf '    - {file: %s, line: %s, target: %s}\n' \
				"$(yaml_escape "${file#$base_dir/}")" "$line" "$(yaml_escape "$symbol")"
		done < <(grep -nE '^[A-Za-z0-9_.%+-]+[[:space:]]*:' "$file" | sed -E 's/^([0-9]+):([^:]+):.*/\1:\2/' || true)
	done < <(find "$base_dir" -name 'Makefile' -type f -print0 | LC_ALL=C sort -z)

	printf '  bash_functions:\n'
	while IFS= read -r -d '' file; do
		while IFS=: read -r line symbol; do
			printf '    - {file: %s, line: %s, function: %s}\n' \
				"$(yaml_escape "${file#$base_dir/}")" "$line" "$(yaml_escape "$symbol")"
		done < <(grep -nE '^[[:alnum:]_]+\(\)[[:space:]]*\{' "$file" | sed -E 's/^([0-9]+):([A-Za-z_][A-Za-z0-9_]*)\(\).*/\1:\2/' || true)
	done < <(find "$base_dir" -name '*.sh' -type f -print0 | LC_ALL=C sort -z)

	printf '  python_definitions:\n'
	while IFS= read -r -d '' file; do
		while IFS=: read -r line symbol; do
			printf '    - {file: %s, line: %s, symbol: %s}\n' \
				"$(yaml_escape "${file#$base_dir/}")" "$line" "$(yaml_escape "$symbol")"
		done < <(grep -nE '^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" | \
				 sed -E 's/^([0-9]+):[[:space:]]*(async[[:space:]]+)?def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1:\3/' || true)
	done < <(find "$base_dir" -name '*.py' -type f -print0 | LC_ALL=C sort -z)
}

# emit_tier2
# ----------
# Emit the Tier 2 scoped module manifest.
emit_tier2() { # 42
	local -r scan_root="$1"
	local -r rel_module="$2"
	local -r include_assets="$3"
	local -r dump_scripts="$4"
	local -r full_path="$scan_root/$rel_module"
	local file rel_file
	local -a text_files=()
	local -a asset_files=()

	[[ -e "$full_path" ]] || fail 1 "module path does not exist: $rel_module"

	while IFS= read -r -d '' file; do
		rel_file="${file#$scan_root/}"
		case "$file" in
			*.png|*.jpg|*.jpeg|*.pdf|*.svg)
				[[ "$include_assets" == true ]] && asset_files+=("$rel_file")
				;;
			*.md|*.py|*.sh|*.tex|*.bib|*.yml|*.yaml|*.json|Makefile)
				text_files+=("$rel_file")
				;;
		esac
	done < <(find "$full_path" -path '*/.*' -prune -o -type f -print0 | LC_ALL=C sort -z)

	cat <<EOF
schema: context-manifest/v2
root: .
tier: 2
module: $(yaml_escape "$rel_module")
authority: {executable: primary, conflicts: human_review}
inventory:
  text: [$(IFS=,; echo "${text_files[*]}")]
  assets: [$(IFS=,; echo "${asset_files[*]}")]
validation:
  targets: $(discover_validate "$full_path")
conflicts: $(scan_conflicts "$scan_root" "$rel_module")
EOF

	if [[ "$dump_scripts" == true ]]; then
		extract_symbols "$full_path"
	fi
}

# main
# ----
# Parse navigation options and emit the requested manifest.
main() { # 65
	local scan_root="$ROOT"
	local module=""
	local include_assets=false
	local scripts=false

	require_command date
	require_command printf

	while [[ $# -gt 0 ]]; do
		case "$1" in
			--root)
				(( $# >= 2 )) || fail 2 "--root requires a directory argument"
				scan_root="$2"
				shift 2
				;;
			--module)
				(( $# >= 2 )) || fail 2 "--module requires a relative path"
				module="$2"
				shift 2
				;;
			--include-assets)
				include_assets=true
				shift
				;;
			--scripts)
				scripts=true
				shift
				;;
			-h|--help)
				fail 2 "navigation help is available through: bash $SCRIPT_DIR/handler.sh --help"
				;;
			--)
				shift
				break
				;;
			*)
				fail 2 "unknown navigate option: $1"
				;;
		esac
	done
	(( $# == 0 )) || fail 2 "unexpected navigate argument: $1"

	local normalized_root
	normalized_root="$(cd -- "$scan_root" 2>/dev/null && pwd -P)" ||
		fail 1 "could not resolve root directory: $scan_root"

	if [[ -z "$module" ]]; then
		[[ "$scripts" == false ]] ||
			fail 2 "--scripts is forbidden in Tier 1 manifest mode (use with --module)"
		emit_tier1 "$normalized_root"
		return 0
	fi

	[[ "$module" != /* ]] || fail 2 "--module must be a relative path: $module"
	local target_path="$normalized_root/$module"
	local canonical_target
	canonical_target="$(cd -- "$target_path" 2>/dev/null && pwd -P)" ||
		canonical_target="$(cd -- "$(dirname -- "$target_path")" 2>/dev/null && pwd -P)/$(basename -- "$target_path")" ||
		fail 2 "cannot resolve module path: $module"
	[[ "$canonical_target" == "$normalized_root" || "$canonical_target" == "$normalized_root"/* ]] ||
		fail 2 "module path escapes root boundary: $module"

	emit_tier2 "$normalized_root" "$module" "$include_assets" "$scripts"
}

main "$@"
