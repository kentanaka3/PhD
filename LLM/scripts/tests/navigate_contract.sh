#!/usr/bin/env bash

# Synthetic public-CLI contract tests for the context manifest navigator.
#
# USAGE
#   bash LLM/scripts/tests/navigate_contract.sh
#
# DESCRIPTION
#   Exercise handler.sh navigate with temporary fixture roots and parse
#   captured manifests using Ruby Psych. No repository data is modified.
#
# SIDE EFFECTS
#   Creates and removes a private temporary directory under TMPDIR.
#
# REPEATABILITY
#   Safe to run repeatedly; fixtures are isolated per invocation.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPTS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly HANDLER="$SCRIPTS_DIR/handler.sh"
readonly REAL_BASH="$(command -v bash)"

source "$SCRIPT_DIR/common.sh"

TEMP_DIR=""
FIXTURE_ROOT=""
OUTSIDE_DIR=""

# Remove the isolated fixture tree when the test exits.
cleanup() { # 5
	if [[ -n "$TEMP_DIR" ]]; then
		rm -rf -- "$TEMP_DIR"
	fi
}

# Assert that a handler navigation command fails without writing stdout.
assert_failure_empty_stdout() { # 9
	local -r stdout_file="$1"
	local -r stderr_file="$2"
	shift 2
	if "$REAL_BASH" "$HANDLER" navigate "$@" >"$stdout_file" 2>"$stderr_file"; then
		fail "expected navigation failure for arguments: $*"
	fi
	[[ ! -s "$stdout_file" ]] || fail "failed navigation wrote partial stdout: $*"
}

# Create the synthetic tree needed for Tier 1 and its outline commands.
prepare_standard_root() { # 9
	mkdir -p -- "$FIXTURE_ROOT/OGS/conf" "$FIXTURE_ROOT/doc" "$FIXTURE_ROOT/LLM"
	printf '# Synthetic OGS\n' > "$FIXTURE_ROOT/OGS/README.md"
	printf '# Synthetic documentation\n' > "$FIXTURE_ROOT/doc/README.md"
	printf '# Synthetic LLM workspace\n' > "$FIXTURE_ROOT/LLM/README.md"
	printf 'root: fixture\n' > "$FIXTURE_ROOT/OGS/conf/config.yaml"
	mkdir -p -- "$FIXTURE_ROOT/odd,[module]" "$FIXTURE_ROOT/empty"
	printf 'synthetic punctuation fixture\n' > "$FIXTURE_ROOT/odd,[module]/name,[part].md"
}

# Check Tier 1 parses and preserves its public collection fields.
test_tier1() { # 19
	local -r output_file="$TEMP_DIR/tier1.yaml"
	"$REAL_BASH" "$HANDLER" navigate --root "$FIXTURE_ROOT" > "$output_file"
	ruby - "$output_file" <<'RUBY'
require "yaml"
manifest = Psych.safe_load(File.read(ARGV.fetch(0)))
raise "Tier 1 schema mismatch" unless manifest.fetch("schema") == "context-manifest/v2"
raise "Tier 1 mismatch" unless manifest.fetch("tier") == 1
raise "modules must be a sequence" unless manifest.fetch("modules").is_a?(Array)
manifest.fetch("modules").each do |mod|
  raise "module entries must be a sequence" unless mod.fetch("entry").is_a?(Array)
end
raise "markdown entries must be a sequence" unless manifest.dig("navigation", "markdown", "entries").is_a?(Array)
raise "yaml entries must be a sequence" unless manifest.dig("navigation", "yaml", "entries").is_a?(Array)
RUBY
	cp -- "$output_file" "$TEMP_DIR/tier1-first.yaml"
	"$REAL_BASH" "$HANDLER" navigate --root "$FIXTURE_ROOT" > "$output_file"
	cmp -s -- "$TEMP_DIR/tier1-first.yaml" "$output_file" || fail "Tier 1 output is not deterministic"
}

# Check Tier 2 parses and round-trips punctuation-bearing module and file paths.
test_punctuation_paths() { # 13
	local -r output_file="$TEMP_DIR/punctuation.yaml"
	"$REAL_BASH" "$HANDLER" navigate --root "$FIXTURE_ROOT" --module 'odd,[module]' > "$output_file"
	ruby - "$output_file" <<'RUBY'
require "yaml"
manifest = Psych.safe_load(File.read(ARGV.fetch(0)))
raise "Tier 2 mismatch" unless manifest.fetch("tier") == 2
raise "module path did not round-trip" unless manifest.fetch("module") == "odd,[module]"
paths = manifest.dig("inventory", "text")
raise "text inventory must be a sequence" unless paths.is_a?(Array)
raise "punctuation path did not round-trip" unless paths.include?("odd,[module]/name,[part].md")
RUBY
}

# Check empty Tier 2 inventory and symbol collections remain YAML sequences.
test_empty_sequences() { # 16
	local -r output_file="$TEMP_DIR/empty.yaml"
	"$REAL_BASH" "$HANDLER" navigate --root "$FIXTURE_ROOT" --module empty --scripts > "$output_file"
	ruby - "$output_file" <<'RUBY'
require "yaml"
manifest = Psych.safe_load(File.read(ARGV.fetch(0)))
inventory = manifest.fetch("inventory")
raise "empty text inventory must be a sequence" unless inventory.fetch("text") == []
raise "empty asset inventory must be a sequence" unless inventory.fetch("assets") == []
symbols = manifest.fetch("symbols")
%w[make_targets bash_functions python_definitions].each do |category|
  raise "#{category} must be a sequence" unless symbols.fetch(category).is_a?(Array)
end
raise "empty symbol groups must remain empty" unless symbols.values.all?(&:empty?)
RUBY
}

# Check absent Tier 1 entry files yield parseable empty navigation sequences.
test_missing_entries() { # 12
	local -r missing_root="$TEMP_DIR/missing-entries"
	local -r output_file="$TEMP_DIR/missing-entries.yaml"
	mkdir -p -- "$missing_root/OGS" "$missing_root/doc" "$missing_root/LLM"
	"$REAL_BASH" "$HANDLER" navigate --root "$missing_root" > "$output_file"
	ruby - "$output_file" <<'RUBY'
require "yaml"
manifest = Psych.safe_load(File.read(ARGV.fetch(0)))
raise "missing markdown entries must be an empty sequence" unless manifest.dig("navigation", "markdown", "entries") == []
raise "missing yaml entries must be an empty sequence" unless manifest.dig("navigation", "yaml", "entries") == []
RUBY
}

# Check empty module values, traversal paths, and external symlinks are rejected.
test_rejections() { # 16
	ln -s -- "$OUTSIDE_DIR" "$FIXTURE_ROOT/external-link"
	ln -s -- "$OUTSIDE_DIR/external.md" "$FIXTURE_ROOT/external-file-link"
	assert_failure_empty_stdout "$TEMP_DIR/empty-module.out" "$TEMP_DIR/empty-module.err" \
		--root "$FIXTURE_ROOT" --module ""
	assert_failure_empty_stdout "$TEMP_DIR/traversal.out" "$TEMP_DIR/traversal.err" \
		--root "$FIXTURE_ROOT" --module ../outside
	assert_failure_empty_stdout "$TEMP_DIR/symlink.out" "$TEMP_DIR/symlink.err" \
		--root "$FIXTURE_ROOT" --module external-link
	assert_failure_empty_stdout "$TEMP_DIR/symlink-file.out" "$TEMP_DIR/symlink-file.err" \
		--root "$FIXTURE_ROOT" --module external-file-link
	mkdir -p -- "$FIXTURE_ROOT/control"
	printf 'unsupported path\n' > "$FIXTURE_ROOT/control/"$'bad\nname.md'
	assert_failure_empty_stdout "$TEMP_DIR/control.out" "$TEMP_DIR/control.err" \
		--root "$FIXTURE_ROOT" --module control
}

# Inject a delegated outline failure and ensure no partial manifest reaches stdout.
test_generation_failure() { # 24
	local -r inject_dir="$TEMP_DIR/injected-bin"
	local -r stdout_file="$TEMP_DIR/generation-failure.out"
	local -r stderr_file="$TEMP_DIR/generation-failure.err"
	mkdir -p -- "$inject_dir"
	cat > "$inject_dir/bash" <<'WRAPPER'
#!/bin/sh
set -eu
for argument in "$@"; do
  case "$argument" in
    */md_nav.sh|*/yaml_nav.sh) exit 73 ;;
  esac
done
exec "$NAVIGATE_TEST_BASH" "$@"
WRAPPER
	chmod 700 -- "$inject_dir/bash"
	if PATH="$inject_dir:$PATH" NAVIGATE_TEST_BASH="$REAL_BASH" "$REAL_BASH" "$HANDLER" navigate \
		--root "$FIXTURE_ROOT" >"$stdout_file" 2>"$stderr_file"; then
		fail "injected outline generation failure unexpectedly succeeded"
	fi
	[[ ! -s "$stdout_file" ]] || fail "generation failure wrote partial stdout"
	[[ "$(<"$stderr_file")" == *"md_nav.sh failed for"* ]] ||
		fail "injected failure did not reach the delegated outline"
}

# Run the isolated public CLI contract checks.
main() { # 22
	require_command ruby
	require_command mktemp
	require_command cmp
	require_command mkdir
	require_command ln
	require_command rm
	require_command chmod
	TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/navigate-contract.XXXXXX")"
	trap cleanup EXIT
	FIXTURE_ROOT="$TEMP_DIR/root"
	OUTSIDE_DIR="$TEMP_DIR/outside"
	mkdir -p -- "$OUTSIDE_DIR"
	prepare_standard_root
	test_tier1
	test_punctuation_paths
	test_empty_sequences
	test_missing_entries
	test_rejections
	test_generation_failure
	printf 'Navigation contract tests passed.\n'
}

main "$@"