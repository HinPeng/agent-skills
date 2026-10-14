#!/usr/bin/env bash
# Generated TEST_FILES run sequentially in separate Python processes.
# Usage: from the intended Linux test working directory:
# bash /path/run_pytest.sh /path/adapted/test/inductor [--dry-run] [-- pytest args...]
set -u
set -o pipefail

TEST_FILES=(
# GENERATED_TEST_FILES
)

TEST_ROOT="${1:?Usage: run_pytest.sh TEST_ROOT [--dry-run] [-- pytest arguments]}"
shift
TEST_ROOT="$(cd "$TEST_ROOT" && pwd)" || exit 2
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")
LOG_DIR="$PWD/logs2"

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
    echo 'Error: TEST_FILES is empty. Use generate_runner.py first.' >&2
    exit 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
    for test_name in "${TEST_FILES[@]}"; do
        [[ -f "$TEST_ROOT/$test_name" ]] || { echo "Missing: $test_name" >&2; exit 2; }
        printf 'DRY RUN: %q -m pytest -vv -rA --tb=short --color=no %q' "$PYTHON_BIN" "$TEST_ROOT/$test_name"
        printf ' %q' "${EXTRA_ARGS[@]}"
        printf '\n'
    done
    exit 0
fi

if [[ "$(uname -s)" != 'Linux' ]]; then
    echo 'Real execution requires Linux Ascend NPU; Windows supports --dry-run only.' >&2
    exit 2
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || exit 127
"$PYTHON_BIN" -m pytest --version || exit 127
if [[ -d "$LOG_DIR" ]] && find "$LOG_DIR" -type f -print -quit | read -r _; then
    echo "Existing logs in $LOG_DIR; choose a fresh run directory to avoid mixing runs." >&2
    exit 2
fi
mkdir -p "$LOG_DIR" || exit 2
overall_rc=0
index=0
for test_name in "${TEST_FILES[@]}"; do
    index=$((index + 1))
    test_file="$TEST_ROOT/$test_name"
    printf -v log_file '%s/%05d.log' "$LOG_DIR" "$index"
    if [[ ! -f "$test_file" ]]; then
        printf 'Test file: %s\nError: test file is missing\n===== exit code: 2 =====\n' "$test_name" > "$log_file"
        overall_rc=2
        continue
    fi
    (
        printf '===== pytest run: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'Test file: %s\nWorking directory: %s\n' "$test_name" "$PWD"
        printf 'Command: %q -m pytest -vv -rA --tb=short --color=no %q' "$PYTHON_BIN" "$test_file"
        printf ' %q' "${EXTRA_ARGS[@]}"
        printf '\n\n'
        "$PYTHON_BIN" -m pytest -vv -rA --tb=short --color=no "$test_file" "${EXTRA_ARGS[@]}"
        pytest_rc=$?
        printf '\n===== exit code: %d =====\n' "$pytest_rc"
        exit "$pytest_rc"
    ) 2>&1 | tee "$log_file"
    pipeline_status=("${PIPESTATUS[@]}")
    rc="${pipeline_status[0]}"
    if [[ "${pipeline_status[1]}" != 0 ]]; then rc=2; fi
    if [[ "$rc" != 0 ]]; then
        overall_rc="$rc"
        printf 'File failed: %s (exit %s); continuing remaining files\n' "$test_name" "$rc" >&2
    fi
done
exit "$overall_rc"
