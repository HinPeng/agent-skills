#!/usr/bin/env bash
set -Eeuo pipefail

UV_BIN="${TORCH_DEV_UV:-uv}"
CPU_INDEX='https://download.pytorch.org/whl/cpu'
if [[ ${TORCH_DEV_UV_CONFIG_TEMPLATE+x} ]]; then
  TEMPLATE="${TORCH_DEV_UV_CONFIG_TEMPLATE}"
else
  # BASH_SOURCE is empty when this worker is streamed with `bash -s` over SSH.
  # Keep template lookup optional so snapshots can use a remote uv.toml (or none).
  script_source="${BASH_SOURCE[0]-}"
  if [[ -n "${script_source}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${script_source}")" && pwd)"
    TEMPLATE="${SCRIPT_DIR}/../references/uv-config.toml"
  else
    SCRIPT_DIR=""
    TEMPLATE=""
  fi
fi

log() { printf '[torch-dev-worker] %s\n' "$*" >&2; }
die() { printf '[torch-dev-worker] ERROR: %s\n' "$*" >&2; exit 1; }

select_config() {
  local selector="$1"
  case "$selector" in
    2_7) ENV_NAME=torch271-py311; TORCH=2.7.1; VISION=0.22.1; AUDIO=2.7.1 ;;
    2_10) ENV_NAME=torch210-py311; TORCH=2.10.0; VISION=0.25.0; AUDIO=2.10.0 ;;
    2_13) ENV_NAME=torch213-py311; TORCH=2.13.0; VISION=0.28.0; AUDIO=2.11.0 ;;
    *) die "unknown selector '$selector'; expected 2_7, 2_10, or 2_13" ;;
  esac
}

assert_safe_path() {
  local path="$1" label="${2:-path}"
  [[ "$path" == /* ]] || die "$label must be an absolute path"
  [[ "$path" != *[[:space:]]* ]] || die "$label contains whitespace"
  [[ "$path" != *[\'\"\`\$\;\|\<\>\&\*\?\[\]\{\}\(\)]* ]] || die "$label contains unsafe shell characters"
}

require_command() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }

python_version() { "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'; }
python_mm() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'; }
python_arch() { "$1" -c 'import platform; print(platform.machine())'; }

torch_version() {
  "$1" -c 'import importlib.metadata as m
try:
    print(m.version("torch"))
except m.PackageNotFoundError:
    import torch
    print(torch.__version__)'
}

torch_base_version() {
  local version
  version="$(torch_version "$1")"
  [[ -n "$version" ]] || die "unable to detect installed torch version"
  printf '%s\n' "${version%%+*}"
}

version_family() {
  local version="$1"
  [[ "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)([.+-].*)?$ ]] || return 1
  printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
}

version_matches_family() {
  local actual="$1" expected="$2" parsed
  parsed="$(version_family "$actual")" || return 1
  [[ "$parsed" == "$expected" ]]
}

config_path() {
  local base="${XDG_CONFIG_HOME:-}"
  if [[ -z "$base" ]]; then
    [[ -n "${HOME:-}" ]] || die 'HOME or XDG_CONFIG_HOME must be set'
    base="${HOME}/.config"
  fi
  [[ "$base" == /* ]] || die 'XDG_CONFIG_HOME or HOME must resolve to an absolute path'
  printf '%s/uv/uv.toml\n' "$base"
}

find_python311() {
  local candidate version
  if [[ -n "${1:-}" && -x "$1" ]]; then
    candidate="$1"
  else
    require_command "$UV_BIN"
    candidate="$($UV_BIN python find 3.11 | tail -n 1)"
  fi
  [[ -x "$candidate" ]] || die "Python 3.11 interpreter not found: $candidate"
  version="$(python_version "$candidate")"
  [[ "$version" == 3.11.15 ]] || die "interpreter must be Python 3.11.15, got $version"
  printf '%s\n' "$candidate"
}

pta_url() {
  local py="$1" torch="$2" mm arch
  mm="$(python_mm "$py")"; arch="$(python_arch "$py")"
  case "$arch" in arm64) arch=aarch64;; amd64) arch=x86_64;; esac
  printf 'http://pytorch-package.obs.cn-north-4.myhuaweicloud.com/cache/v%s/pytorchv%s_%s_%s.tar.gz\n' "$torch" "$torch" "$mm" "$arch"
}

print_config() {
  (($# == 1)) || die 'usage: --print-config SELECTOR'
  select_config "$1"
  printf '%s|%s|%s|%s|%s\n' "$1" "$ENV_NAME" "$TORCH" "$VISION" "$AUDIO"
}

new_evidence_dir() {
  local env_root="$1" name="$2" base ts
  base="${env_root}/.pta-evidence"; mkdir -p "$base"
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  mktemp -d "${base}/${name}-${ts}-XXXXXX"
}

write_common_evidence() {
  local dir="$1" selector="$2" env_root="$3" py="$4" url="$5"
  printf '%s\n' "$url" >"$dir/pta-url.txt"
  printf '%s\n' "$(date -u +%FT%TZ)" >"$dir/timestamp.txt"
  {
    printf 'selector=%s\nenv_root=%s\nenv_name=%s\ninterpreter=%s\npython_version=%s\n' "$selector" "$env_root" "$ENV_NAME" "$py" "$(python_version "$py")"
    printf 'pta_url=%s\n' "$url"
  } >"$dir/config.txt"
}

install_latest_one() {
  local env_root="$1" selector="$2" env_dir py url evidence archive extract wheel sha expected actual metadata_version arch installed_torch installed_base
  select_config "$selector"; assert_safe_path "$env_root" ENV_ROOT
  env_dir="${env_root}/${ENV_NAME}"; [[ -d "$env_dir" ]] || die "environment not found: $env_dir"
  py="${env_dir}/bin/python"; [[ -x "$py" ]] || die "environment interpreter not found: $py"
  [[ "$(python_version "$py")" == 3.11.15 ]] || die 'environment interpreter must be Python 3.11.15'
  installed_torch="$(torch_version "$py")" || die "unable to detect installed torch version from $py"
  installed_base="${installed_torch%%+*}"
  [[ "$installed_base" == "$TORCH" ]] || die "installed torch version $installed_torch does not match selector $TORCH"
  url="$(pta_url "$py" "$installed_base")"; evidence="$(new_evidence_dir "$env_root" "$ENV_NAME")"
  arch="$(python_arch "$py")"; case "$arch" in arm64) arch=aarch64;; amd64) arch=x86_64;; esac
  write_common_evidence "$evidence" "$selector" "$env_root" "$py" "$url"
  archive="$evidence/pta-package.tar.gz"; extract="$evidence/extracted"
  require_command tar
  local metadata_python="${TORCH_DEV_METADATA_PYTHON:-python3}"
  require_command "$metadata_python"
  local curl_bin="${TORCH_DEV_CURL:-curl}"; require_command "$curl_bin"
  log "downloading PTA package for $ENV_NAME"
  "$curl_bin" -fL -D "$evidence/http-headers.txt" -o "$archive" "$url"
  mkdir -p "$extract"; tar -xzf "$archive" -C "$extract"
  mapfile -t wheels < <(find "$extract" -type f -name 'torch_npu-*.whl' -print | sort)
  ((${#wheels[@]} == 1)) || die "expected exactly one torch_npu wheel; evidence: $evidence"
  wheel="${wheels[0]}"; sha="${wheel}.sha256"
  [[ -f "$sha" ]] || die "checksum file not found; evidence: $evidence"
  mapfile -t sha_files < <(find "$extract" -type f -name '*.sha256' -print | sort)
  ((${#sha_files[@]} == 1)) || die "expected exactly one .sha256 file; evidence: $evidence"
  [[ "${sha_files[0]}" == "$sha" ]] || die "checksum file does not match wheel; evidence: $evidence"
  expected="$(awk 'NR==1 {print $1}' "$sha")"
  [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || die "invalid expected checksum; evidence: $evidence"
  local sha_bin="${TORCH_DEV_SHA256SUM:-sha256sum}"; require_command "$sha_bin"
  actual="$("$sha_bin" "$wheel" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "checksum mismatch; evidence: $evidence"
  [[ "$(basename "$wheel")" =~ ^torch_npu-[^-]+-cp311-cp311-(linux|manylinux_[0-9]+_[0-9]+)_${arch}\.whl$ ]] || die "wheel filename must target cp311 and host architecture; evidence: $evidence"
  metadata_version="$($metadata_python - "$wheel" <<'PYMETA'
import sys, zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith("/METADATA"):
            for line in archive.read(name).decode("utf-8", "replace").splitlines():
                if line.startswith("Version:"):
                    print(line.split(":", 1)[1].strip())
                    raise SystemExit
raise SystemExit("METADATA Version field not found")
PYMETA
)"
  version_matches_family "$metadata_version" "$TORCH" || die "torch_npu metadata version $metadata_version does not match torch family $TORCH; evidence: $evidence"
  "$UV_BIN" pip install --python "$py" --force-reinstall --no-deps "$wheel"
  printf '%s\n' "$(basename "$wheel")" >"$evidence/wheel-filename.txt"
  printf '%s\n' "$metadata_version" >"$evidence/wheel-version.txt"
  printf '%s  %s\n' "$actual" "$(basename "$wheel")" >"$evidence/wheel.sha256"
  verify_one "$env_root" "$selector"
  "$UV_BIN" pip freeze --python "$py" >"$evidence/uv-pip-freeze.txt"
}

create_one() {
  local env_root="$1" selector="$2" env_dir py
  select_config "$selector"; assert_safe_path "$env_root" ENV_ROOT
  env_dir="${env_root}/${ENV_NAME}"; [[ ! -e "$env_dir" ]] || die "environment already exists: $env_dir"
  require_command "$UV_BIN"; mkdir -p "$env_root"
  py="$(find_python311)"
  "$UV_BIN" venv --python "$py" "$env_dir"
  py="${env_dir}/bin/python"; "$UV_BIN" pip install --python "$py" --index-url "$CPU_INDEX" "torch==${TORCH}" "torchvision==${VISION}" "torchaudio==${AUDIO}"
  install_latest_one "$env_root" "$selector"
}

verify_one() {
  local env_root="$1" selector="$2" env_dir py safe installed_torch installed_base
  select_config "$selector"; assert_safe_path "$env_root" ENV_ROOT; env_dir="${env_root}/${ENV_NAME}"; py="${env_dir}/bin/python"; [[ -x "$py" ]] || die "environment interpreter not found: $py"
  [[ "$(python_version "$py")" == 3.11.15 ]] || die 'environment interpreter must be Python 3.11.15'
  installed_torch="$(torch_version "$py")" || die "unable to detect installed torch version from $py"
  installed_base="${installed_torch%%+*}"
  [[ "$installed_base" == "$TORCH" ]] || die "installed torch version $installed_torch does not match selector $TORCH"
  safe="$(mktemp -d /tmp/torch-dev-env-verify-XXXXXX)"
  (cd "$safe" && "$py" - "$installed_torch" "$VISION" "$AUDIO" "$installed_base" <<'PY'
import sys
import torch, torchvision, torchaudio, torch_npu
t, v, a, base = sys.argv[1:]
assert sys.version_info[:3] == (3, 11, 15)
assert torch.__version__ == t
assert torchvision.__version__ == v + '+cpu'
assert torchaudio.__version__ == a + '+cpu'
assert tuple(int(x) for x in __import__("re").match(r"^(\d+)\.(\d+)\.(\d+)", torch_npu.__version__).groups()) == tuple(int(x) for x in __import__("re").match(r"^(\d+)\.(\d+)\.(\d+)", base).groups())
assert (torch.tensor([1, 2]) + 1).tolist() == [2, 3]
print(sys.version.split()[0], torch.__version__, torchvision.__version__, torchaudio.__version__, torch_npu.__version__)
PY
  )
  "$UV_BIN" pip check --python "$py"
}

snapshot_one() {
  local env_root="$1" output="$2" selector="$3" env_dir py name url sources_file evidence_dir evidence_url f
  local -a direct_urls
  select_config "$selector"; name="$ENV_NAME"; env_dir="${env_root}/${name}"; py="${env_dir}/bin/python"; [[ -x "$py" ]] || die "environment interpreter not found: $py"
  "$UV_BIN" pip freeze --python "$py" >"$output/${name}.freeze.txt"
  sources_file="$output/${name}.sources.md"
  evidence_dir=''
  if [[ -d "$env_root/.pta-evidence" ]]; then
    evidence_dir="$(find "$env_root/.pta-evidence" -mindepth 1 -maxdepth 1 -type d -name "${name}-*" -print | sort | tail -n 1)"
  fi
  url="$(pta_url "$py" "$TORCH" 2>/dev/null || true)"
  {
    printf '# %s\n\n- host/local mode: %s\n- env root: %s\n- env path: %s\n- interpreter: %s\n- snapshot time: %s\n' "$name" "${TORCH_DEV_HOST_MODE:-local}" "$env_root" "$env_dir" "$py" "$(date -u +%FT%TZ)"
    local config_file
    config_file="$(config_path)"
    if [[ -f "$config_file" ]]; then
      printf '%s\n' '- active uv.toml:'
      sed 's/^/    /' "$config_file"
    elif [[ -f "$env_root/uv.toml" ]]; then
      printf -- '- active uv.toml: %s\n' "$env_root/uv.toml"
      sed 's/^/    /' "$env_root/uv.toml"
    elif [[ -f "$TEMPLATE" ]]; then
      printf -- '- active uv.toml template: %s\n' "$TEMPLATE"
      sed 's/^/    /' "$TEMPLATE"
    else
      printf '%s\n' '- active uv.toml: none found'
    fi
    printf '%s\n' "- PyTorch CPU index: $CPU_INDEX"
    [[ -n "$url" ]] && printf '%s\n' "- PTA URL: $url"
    printf '%s\n' '## PTA wheel evidence'
    if [[ -n "$evidence_dir" ]]; then
      evidence_url=''; [[ -f "$evidence_dir/pta-url.txt" ]] && evidence_url="$(<"$evidence_dir/pta-url.txt")"
      [[ -n "$evidence_url" ]] && printf '%s\n' "- PTA URL (evidence): $evidence_url"
      [[ -f "$evidence_dir/wheel-filename.txt" ]] && printf '%s\n' "- Wheel filename: $(<"$evidence_dir/wheel-filename.txt")"
      [[ -f "$evidence_dir/wheel-version.txt" ]] && printf '%s\n' "- Wheel version: $(<"$evidence_dir/wheel-version.txt")"
      if [[ -f "$evidence_dir/wheel.sha256" ]]; then
        printf '%s\n' "- Wheel SHA256: $(awk 'NR == 1 {print $1}' "$evidence_dir/wheel.sha256")"
      fi
    else
      printf '%s\n' '- No PTA installation evidence found.'
    fi
    printf '%s\n' '## Direct wheel provenance'
    printf '%s\n' 'Installed wheel provenance is recorded from direct_url.json; no inference is made from historical package sources.'
    mapfile -t direct_urls < <(find "$env_dir" -type f -name direct_url.json -print | sort)
    if ((${#direct_urls[@]} == 0)); then
      printf '%s\n' '- direct_url.json: none found'
    else
      for f in "${direct_urls[@]}"; do
        printf '\n### %s\n\n~~~json\n' "$f"
        sed -n '1,$p' "$f"
        printf '\n~~~\n'
      done
    fi
    printf '%s\n' '- Ordinary historical package sources are not traceable from freeze output alone.'
  } >"$sources_file"
}

main() {
  [[ $# -gt 0 ]] || die 'usage: --print-config SELECTOR | create ENV_ROOT SELECTOR... | install-latest-torch-npu ENV_ROOT SELECTOR... | verify ENV_ROOT SELECTOR... | snapshot-dependencies ENV_ROOT OUTPUT_DIR'
  if [[ "$1" == --print-config ]]; then shift; print_config "$@"; return; fi
  local command="$1"; shift
  case "$command" in
    create|install-latest-torch-npu|verify)
      (($# >= 2)) || die "$command requires ENV_ROOT and one or more selectors"
      local root="$1"; shift; assert_safe_path "$root" ENV_ROOT
      local selector; for selector in "$@"; do case "$command" in create) create_one "$root" "$selector";; install-latest-torch-npu) install_latest_one "$root" "$selector";; verify) verify_one "$root" "$selector";; esac; done ;;
    snapshot-dependencies)
      (($# == 2)) || die 'snapshot-dependencies requires ENV_ROOT OUTPUT_DIR'; local root="$1" output="$2"; assert_safe_path "$root" ENV_ROOT; assert_safe_path "$output" OUTPUT_DIR; mkdir -p "$output"; rm -f -- "$output/sources.md"; for selector in 2_7 2_10 2_13; do snapshot_one "$root" "$output" "$selector"; done ;;
    *) die "unknown command: $command" ;;
  esac
}

main "$@"
