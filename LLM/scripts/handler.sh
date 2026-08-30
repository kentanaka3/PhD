#!/usr/bin/env bash

# LLM Markdown handler
# =======================
#
# Coordinate the LLM workspace lifecycle from one command-line interface.
# The handler combines workspace initialization, Markdown navigation reporting,
# and workspace validation without changing the behavior of the original
# standalone scripts.
#
# USAGE
#   bash handler.sh init [--dry-run]
#   bash handler.sh navigate [--root DIRECTORY|--file FILE] [--format table|tsv] [--scripts]
#   bash handler.sh validate [--root DIRECTORY|--file FILE]
#   bash handler.sh --help
#
# COMMANDS
#   init       Create missing LLM directories and starter documents.
#   navigate   Print a Markdown navigation report to standard output.
#   validate   Validate documentation, links, paths, and script syntax.
#
# Function          | description
# ------------------|--------------------------------------------------------
# log               | Print a timestamped informational message.
# fail              | Report an error and terminate.
# require_command   | Verify that a required executable is available.
# usage             | Print command-line usage information.
# initialize        | Create missing workspace directories and starters.
# navigate          | Generate the Markdown navigation report.
# validate          | Check required files, links, paths, and script syntax.
# navigate_scripts  | Inventory Makefile targets, Bash functions, and Python
#                     definitions with line-number references.
# validate_functions| Validate Bash function documentation and line counts.
# main              | Parse the command and dispatch the requested operation.
#
# SAFETY
#   `init` is idempotent and never overwrites existing starter files. Use
#   `--dry-run` to report changes without modifying the workspace.
#   `navigate` writes only to standard output.
#   `validate` does not modify files or contact external services.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly ROOT="$(cd -- "$WORKSPACE/.." && pwd)"

# log
# ---
# Print a timestamped informational message to standard error.
log() { # 6
    printf '[%s][%s] %s\n' \
        "$SCRIPT" \
        "$(date '+%Y-%m-%d %H:%M:%S%z')" \
        "$*" >&2
}

# fail
# ----
# Report an error message and terminate the script with a non-zero exit code.
fail() { # 7
    local -r status="$1"
    shift

    log "ERROR: $*"
    exit "$status"
}

# require_command
# ---------------
# Verify that a required executable is available.
require_command() { # 6
    local -r command_name="$1"

    command -v "$command_name" >/dev/null 2>&1 ||
        fail 127 "required command not found: $command_name"
}

# usage
# -----
# Print command-line usage information.
usage() { # 26
    cat <<EOF
Usage: $SCRIPT COMMAND [OPTIONS]

Commands:
  init [--dry-run]       Create missing workspace directories and starters.
  navigate [OPTIONS]    Generate a Markdown navigation report.
  validate [--root DIR]  Validate documentation and project scripts.

Options:
  -h, --help            Show this help message and exit.
  --                    End option processing.

Navigate options:
  --root DIRECTORY      Scan DIRECTORY instead of the PhD project root.
  --file FILE           Navigate or validate one supported script file.
  --format FORMAT       Use table (default) or machine-readable tsv output.
  --scripts             Include Makefile, Bash, and Python symbol inventory.

Examples:
  bash $SCRIPT init
  bash $SCRIPT navigate
  bash $SCRIPT navigate --root /path/to/PhD > markdown_navigation_report.md
  bash $SCRIPT validate
EOF
}

# initialize
# ----------
# Create missing workspace directories and starter documents.
initialize() { # 37
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
        if [[ "$dry_run" == false ]]; then
            mkdir -p -- "$WORKSPACE/$directory"
        else
            log "Would create directory: $WORKSPACE/$directory"
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

# navigation_rows
# ---------------
# Generate the Markdown navigation report.
navigation_rows() { # 34
    local -r scan_root="$1"
    local markdown_count file relative lines headings

    require_command awk
    require_command find
    require_command sort
    require_command wc
    [[ -d "$scan_root" ]] ||
        fail 1 "scan root does not exist or is not a directory: $scan_root"

    markdown_count=$(find "$scan_root" -path "$scan_root/.git" -prune -o \
        -type f -name '*.md' -print | wc -l)
    printf '# Markdown navigation report\n\nRepository: `%s`\n\n' "$scan_root"
    printf 'Markdown files discovered (excluding `.git`): `%s`\n\n' "$markdown_count"
    printf '| File | Lines | Headings |\n|---|---:|---|\n'
    while IFS= read -r -d '' file; do
        relative=${file#"$scan_root/"}
        lines=$(wc -l < "$file")
        headings=$(awk -v file="$relative" '
            /^```|^~~~/ { fenced = !fenced; next }
            !fenced && /^#{1,6}[[:space:]]/ {
                gsub(/\|/, "\\|", $0)
                printf "%s:%d %s; ", file, FNR, $0
            }
        ' "$file")
        [[ -n "$headings" ]] || headings='(no Markdown heading found)'
        headings=${headings//|/\\|}
        printf '| `%s` | %s | %s |\n' "$relative" "$lines" "$headings"
    done < <(
        find "$scan_root" -path "$scan_root/.git" -prune -o \
            -type f -name '*.md' -print0 | sort -z
    )
}

# navigation_tsv
# --------------
# Generate the tab-separated navigation report.
navigation_tsv() { # 20
    local -r scan_root="$1"
    local file relative lines headings

    printf 'file\tlines\theadings\n'
    while IFS= read -r -d '' file; do
        relative=${file#"$scan_root/"}
        lines=$(wc -l < "$file")
        headings=$(awk '
            /^```|^~~~/ { fenced = !fenced; next }
            !fenced && /^#{1,6}[[:space:]]/ { printf "%s; ", $0 }
        ' "$file")
        headings=${headings//$'\t'/ }
        headings=${headings//$'\n'/ }
        printf '%s\t%s\t%s\n' "$relative" "$lines" "${headings:-"(no heading)"}"
    done < <(
        find "$scan_root" -path "$scan_root/.git" -prune -o \
            -type f -name '*.md' -print0 | sort -z
    )
}

# validate_markdown
# -----------------
# Validate local Markdown paths and fenced code blocks.
validate_markdown() { # 25
    local -r scan_root="$1"
    local failures=0 file target link line_number content

    while IFS= read -r -d '' file; do
        if (( $(awk '/^```|^~~~/ { fenced = !fenced } END { print fenced + 0 }' "$file") != 0 )); then
            printf 'ERR %s (unclosed fenced code block)\n' "${file#$scan_root/}" >&2
            ((failures += 1))
        fi
        while IFS=: read -r line_number content; do
            link=$(printf '%s\n' "$content" | sed -E 's/.*\]\(([^)#]+).*/\1/')
            [[ -n "$link" && "$link" != http* && "$link" != mailto:* ]] || continue
            target="$scan_root/$(dirname -- "${file#$scan_root/}")/$link"
            [[ -e "$target" ]] || {
                printf 'ERR %s:%s (broken Markdown path: %s)\n' \
                    "${file#$scan_root/}" "$line_number" "$link" >&2
                ((failures += 1))
            }
        done < <(grep --line-number --extended-regexp '\]\([^)#]+(\#[^)]*)?\)' "$file" || true)
    done < <(
        find "$scan_root" -path "$scan_root/.git" -prune -o \
            -type f -name '*.md' -print0 | sort -z
    )
    return "$failures"
}

# validate_script_references
# --------------------------
# Detect stale references to project shell scripts.
validate_script_references() { # 20
    local -r scan_root="$1"
    local failures=0 file reference

    while IFS= read -r -d '' file; do
        while IFS= read -r reference; do
            [[ -e "$scan_root/$reference" ]] || {
                printf 'ERR %s (stale or missing script reference: %s)\n' \
                    "${file#$scan_root/}" "$reference" >&2
                ((failures += 1))
            }
        done < <(grep --only-matching --no-filename \
            --extended-regexp '(LLM/scripts|OGS/utils/Leonardo|OGS/utils/\.local)/[A-Za-z0-9_.-]+\.sh' \
            "$file" | sort -u || true)
    done < <(
        find "$scan_root" -path "$scan_root/.git" -prune -o \
            -type f -name '*.md' -print0 | sort -z
    )
    return "$failures"
}

# validate_bash
# ------------
# Validate the syntax of every project Bash script.
validate_bash() { # 16
    local -r scan_root="$1"
    local failures=0 file

    require_command bash
    while IFS= read -r -d '' file; do
        if ! bash -n "$file"; then
            printf 'ERR %s (Bash syntax check failed)\n' "${file#$scan_root/}" >&2
            ((failures += 1))
        fi
    done < <(
        find "$scan_root" -path "$scan_root/.git" -prune -o \
            -type f -name '*.sh' -print0 | sort -z
    )
    return "$failures"
}

# Validate the required workspace, Markdown, and Bash surfaces.
# navigate_scripts
# ----------------
# Inventory Makefile targets, Bash functions, and Python definitions.
navigate_scripts() { # 44
    local -r scan_root="$1"
    local -r selected_file="${2:-}"
    local file line symbol

    printf '# Script navigation report\n\nRoot: `%s`\n\n' "$scan_root"
    printf '| File | Type | Line | Symbol |\n|---|---|---:|---|\n'
    while IFS= read -r -d '' file; do
        case "$(basename -- "$file")" in
            Makefile)
                while IFS=: read -r line symbol; do
                    printf '| `%s` | Make target | %s | `%s` |\n' \
                        "${file#$scan_root/}" "$line" "$symbol"
                done < <(grep --line-number --extended-regexp \
                    '^[[:alnum:]_.%+-]+[[:space:]]*:' "$file" |
                    sed -E 's/^([0-9]+):([^:]+):.*/\1:\2/' || true)
                ;;
            *.sh)
                while IFS=: read -r line symbol; do
                    printf '| `%s` | Bash function | %s | `%s` |\n' \
                        "${file#$scan_root/}" "$line" "$symbol"
                done < <(grep --line-number --extended-regexp \
                    '^[[:alnum:]_]+\(\)[[:space:]]*\{' "$file" |
                    sed -E 's/^([0-9]+):([A-Za-z_][A-Za-z0-9_]*)\\(\\).*/\1:\2/' || true)
                ;;
            *.py)
                while IFS=: read -r line symbol; do
                    printf '| `%s` | Python definition | %s | `%s` |\n' \
                        "${file#$scan_root/}" "$line" "$symbol"
                done < <(grep --line-number --extended-regexp \
                    '^[[:space:]]*(async[[:space:]]+)?def[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" |
                    sed -E 's/^([0-9]+):[[:space:]]*(async[[:space:]]+)?def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1:\3/' || true)
                ;;
        esac
    done < <(
        if [[ -n "$selected_file" ]]; then
            printf '%s\0' "$selected_file"
        else
            find "$scan_root" -path "$scan_root/.git" -prune -o \
                \( -name 'Makefile' -o -name '*.sh' -o -name '*.py' \) \
                -type f -print0 | sort -z
        fi
    )
}

# validate_functions
# ------------------
# Validate inline Bash function line counts in this handler.
validate_functions() { # 32
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
            actual=$(awk -v start="$line" '
                NR > start && /^}$/ {
                    print NR - start + 1
                    exit
                }
            ' "$file")
            if [[ "$actual" != "$count" ]]; then
                printf 'ERR %s:%s (documented %s lines, counted %s)\n' \
                    "${file#$scan_root/}" "$line" "$count" "${actual:-unknown}" >&2
                ((failures += 1))
            fi
        done < <(grep --line-number --extended-regexp \
            '^[[:alnum:]_]+\(\)[[:space:]]*\{' "$file" || true)
    done < <(
        printf '%s\0' "$target_file"
    )
    return "$failures"
}

# validate_single_file
# --------------------
# Validate one supported Bash script or Markdown file.
validate_single_file() { # 49
    local -r scan_root="$ROOT"
    local -r file="$1"
    local failures=0
    local target link line_number content reference

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
            while IFS=: read -r line_number content; do
                link=$(printf '%s\n' "$content" | sed -E 's/.*\]\(([^)#]+).*/\1/')
                [[ -n "$link" && "$link" != http* && "$link" != mailto:* ]] || continue
                target="$(dirname -- "$file")/$link"
                [[ -e "$target" ]] || {
                    printf 'ERR %s:%s (broken Markdown path: %s)\n' \
                        "$file" "$line_number" "$link" >&2
                    ((failures += 1))
                }
            done < <(grep --line-number --extended-regexp '\]\([^)#]+(\#[^)]*)?\)' "$file" || true)
            while IFS= read -r reference; do
                [[ -e "$scan_root/$reference" ]] || {
                    printf 'ERR %s (stale or missing script reference: %s)\n' \
                        "$file" "$reference" >&2
                    ((failures += 1))
                }
            done < <(grep --only-matching --no-filename \
                --extended-regexp '(LLM/scripts|OGS/utils/Leonardo|OGS/utils/\.local)/[A-Za-z0-9_.-]+\.sh' \
                "$file" | sort -u || true)
            ;;
        *)
            fail 2 "single-file validation supports Bash scripts (*.sh) and Markdown (*.md): $file"
            ;;
    esac

    (( failures == 0 )) ||
        fail 1 "single-file validation failed with $failures issue(s)"
    printf 'Validation passed: %s\n' "$file"
}

validate() { # 43
    local -r scan_root="$1"
    local -a required=(
        "$scan_root/AGENTS.md"
        "$scan_root/LLM/README.md"
        "$scan_root/LLM/00_governance/scope.md"
        "$scan_root/LLM/00_governance/decisions.md"
        "$scan_root/LLM/00_governance/open_questions.md"
        "$scan_root/LLM/01_context/pipeline_map.md"
        "$scan_root/LLM/01_context/glossary.md"
        "$scan_root/LLM/02_prompts/code_review.md"
        "$scan_root/LLM/03_workflows/experiment_lifecycle.md"
        "$scan_root/LLM/05_evaluations/rubric.md"
        "$scan_root/LLM/templates/experiment.md"
        "$scan_root/LLM/.gitignore"
        "$scan_root/LLM/scripts/handler.sh"
    )
    local failures=0 file

    for file in "${required[@]}"; do
        if [[ -s "$file" ]]; then
            printf 'OK  %s\n' "${file#$scan_root/}"
        else
            printf 'ERR %s (missing or empty)\n' "${file#$scan_root/}" >&2
            ((failures += 1))
        fi
    done

    if [[ ! -x "$scan_root/LLM/scripts/handler.sh" ]]; then
        printf 'ERR LLM/scripts/handler.sh is not executable\n' >&2
        ((failures += 1))
    fi

    validate_markdown "$scan_root" || failures=$((failures + $?))
    validate_script_references "$scan_root" || failures=$((failures + $?))
    validate_bash "$scan_root" || failures=$((failures + $?))
    validate_functions "$scan_root/LLM/scripts/handler.sh" ||
        failures=$((failures + $?))

    (( failures == 0 )) ||
        fail 1 "validation failed with $failures issue(s)"
    printf '\nValidation passed: %d required files checked.\n' "${#required[@]}"
}

# main
# ----
# Parse the command and dispatch the requested operation.
main() { # 120
    local command="${1:-}"
    local scan_root="$ROOT"
    local file_path=
    local dry_run=false
    local format=table

    require_command date
    require_command printf

    case "$command" in
        init)
            shift
            if [[ "${1:-}" == "--dry-run" ]]; then
                dry_run=true
                shift
            fi
            (( $# == 0 )) || fail 2 "unknown init option: $1"
            initialize "$dry_run"
            ;;
        navigate)
            shift
            local scripts=false
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
                    --format)
                        (( $# >= 2 )) || fail 2 "--format requires a value"
                        format="$2"
                        shift 2
                        ;;
                    --scripts)
                        scripts=true
                        shift
                        ;;
                    -h|--help)
                        usage
                        return 0
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
            if [[ -n "$file_path" ]]; then
                file_path="$(cd -- "$(dirname -- "$file_path")" 2>/dev/null && pwd)/$(basename -- "$file_path")" ||
                    fail 1 "could not resolve file: $file_path"
                [[ -f "$file_path" ]] || fail 1 "file does not exist: $file_path"
                navigate_scripts "$(dirname -- "$file_path")" "$file_path"
            else
                local requested_root="$scan_root"
                scan_root="$(cd -- "$requested_root" 2>/dev/null && pwd)" ||
                    fail 1 "could not resolve navigation root: $requested_root"
                case "$format" in
                    table) navigation_rows "$scan_root" ;;
                    tsv) navigation_tsv "$scan_root" ;;
                    *) fail 2 "unsupported navigation format: $format" ;;
                esac
            fi
            if [[ "$scripts" == true ]]; then
                [[ -n "$file_path" ]] || navigate_scripts "$scan_root"
            fi
            ;;
        validate)
            shift
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
                    -h|--help) usage; return 0 ;;
                    *) fail 2 "unknown validate option: $1" ;;
                esac
            done
            if [[ -n "$file_path" ]]; then
                local requested_file="$file_path"
                file_path="$(cd -- "$(dirname -- "$requested_file")" 2>/dev/null && pwd)/$(basename -- "$requested_file")" ||
                    fail 1 "could not resolve file: $requested_file"
                validate_single_file "$file_path"
            else
                local requested_root="$scan_root"
                scan_root="$(cd -- "$requested_root" 2>/dev/null && pwd)" ||
                    fail 1 "could not resolve validation root: $requested_root"
                validate "$scan_root"
            fi
            ;;
        -h|--help)
            usage
            ;;
        "")
            usage >&2
            fail 2 "a command is required"
            ;;
        *)
            usage >&2
            fail 2 "unknown command: $command"
            ;;
    esac
}

main "$@"
