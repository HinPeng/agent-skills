#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${TORCH_DEV_WORKER:-${SCRIPT_DIR}/torch-env-worker.sh}"
TEMPLATE="${TORCH_DEV_UV_CONFIG_TEMPLATE:-${SCRIPT_DIR}/../references/uv-config.toml}"
SSH_BIN="${TORCH_DEV_SSH:-ssh}"

die() { printf '[torch-dev-env] ERROR: %s\n' "$*" >&2; exit 2; }
log() { printf '[torch-dev-env] %s\n' "$*" >&2; }

assert_safe_root() {
  local value="$1"
  [[ "$value" == /* ]] || die 'env-root must be an absolute path'
  [[ "$value" != *[[:space:]]* ]] || die 'env-root contains whitespace'
  [[ "$value" != *[\'\"\`\$\;\|\<\>\&\*\?\[\]\{\}\(\)]* ]] || die 'env-root contains unsafe shell characters'
}

assert_safe_host() {
  local value="$1"
  [[ -z "$value" || "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'host contains unsafe characters'
}

select_valid() {
  case "$1" in 2_7|2_10|2_13) ;; *) die "unknown selector '$1'; expected 2_7, 2_10, or 2_13" ;; esac
}

config_path_local() {
  local base="${XDG_CONFIG_HOME:-}"
  if [[ -z "$base" ]]; then
    [[ -n "${HOME:-}" ]] || die 'HOME or XDG_CONFIG_HOME must be set'
    base="${HOME}/.config"
  fi
  [[ "$base" == /* ]] || die 'XDG_CONFIG_HOME or HOME must resolve to an absolute path'
  printf '%s/uv/uv.toml\n' "$base"
}

show_sync_preview() {
  local host="$1" target template_q
  [[ -f "$TEMPLATE" ]] || die "uv config template not found: $TEMPLATE"
  if [[ -z "$host" ]]; then
    target="$(config_path_local)"
    log "target mode: local"
    log "target host: local"
    log "uv config path: $target"
    if [[ -f "$target" ]]; then
      if cmp -s "$target" "$TEMPLATE"; then
        log 'uv config status: identical to template'
      else
        log 'uv config status: differs; proposed changes:'
        diff -u "$target" "$TEMPLATE" >&2 || true
      fi
    else
      log 'uv config status: absent; proposed file is the template'
    fi
    return
  fi
  log 'target mode: remote'
  log "target host: $host"
  template_q="$(printf '%q' "$(<"$TEMPLATE")")"
  "$SSH_BIN" -T "$host" bash -s -- <<REMOTE_PREVIEW
set -Eeuo pipefail
base="\${XDG_CONFIG_HOME:-}"
if [[ -z "\$base" ]]; then
  [[ -n "\${HOME:-}" ]] || { printf '%s\n' '[torch-dev-env] ERROR: HOME or XDG_CONFIG_HOME must be set' >&2; exit 2; }
  base="\$HOME/.config"
fi
[[ "\$base" == /* ]] || { printf '%s\n' '[torch-dev-env] ERROR: remote config base must be absolute' >&2; exit 2; }
target="\$base/uv/uv.toml"
printf '[torch-dev-env] uv config path: %s\n' "\$target" >&2
template=${template_q}
if [[ -f "\$target" ]]; then
  if cmp -s "\$target" <(printf '%s\n' "\$template"); then
    printf '%s\n' '[torch-dev-env] uv config status: identical to template' >&2
  else
    printf '%s\n' '[torch-dev-env] uv config status: differs; proposed changes:' >&2
    diff -u "\$target" <(printf '%s\n' "\$template") >&2 || true
  fi
else
  printf '%s\n' '[torch-dev-env] uv config status: absent; proposed file is the template' >&2
fi
REMOTE_PREVIEW
}

sync_local() {
  local target dir tmp backup stamp
  [[ -f "$TEMPLATE" ]] || die "uv config template not found: $TEMPLATE"
  target="$(config_path_local)"; dir="$(dirname "$target")"
  if [[ -f "$target" ]] && cmp -s "$target" "$TEMPLATE"; then
    log "uv config already matches template: $target"; return
  fi
  if [[ -f "$target" ]]; then
    log 'uv config differs; proposed changes:'
    diff -u "$target" "$TEMPLATE" >&2 || true
  fi
  mkdir -p "$dir"; umask 077
  if [[ -f "$target" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"; backup="${target}.bak.${stamp}"
    [[ ! -e "$backup" ]] || backup="${target}.bak.${stamp}.$$"
    cp -p -- "$target" "$backup"; chmod 600 -- "$backup"
    log "backed up existing uv config to $backup"
  fi
  tmp="$(mktemp "${dir}/.uv.toml.tmp.XXXXXX")"
  cp -- "$TEMPLATE" "$tmp"; chmod 600 -- "$tmp"; mv -f -- "$tmp" "$target"
  log "synchronized uv config: $target"
}

sync_remote() {
  local host="$1" template_q
  [[ -f "$TEMPLATE" ]] || die "uv config template not found: $TEMPLATE"
  template_q="$(printf '%q' "$(<"$TEMPLATE")")"
  "$SSH_BIN" -T "$host" bash -s -- <<REMOTE_SYNC
set -Eeuo pipefail
template=${template_q}
base="\${XDG_CONFIG_HOME:-}"
if [[ -z "\$base" ]]; then
  [[ -n "\${HOME:-}" ]] || { printf '%s\n' '[torch-dev-env] ERROR: HOME or XDG_CONFIG_HOME must be set' >&2; exit 2; }
  base="\$HOME/.config"
fi
[[ "\$base" == /* ]] || { printf '%s\n' '[torch-dev-env] ERROR: remote config base must be absolute' >&2; exit 2; }
target="\$base/uv/uv.toml"
dir="\$(dirname "\$target")"; mkdir -p "\$dir"
if [[ -f "\$target" ]] && cmp -s "\$target" <(printf '%s\\n' "\$template"); then exit 0; fi
if [[ -f "\$target" ]]; then
  diff -u "\$target" <(printf '%s\\n' "\$template") >&2 || true
  stamp="\$(date -u +%Y%m%dT%H%M%SZ)"; backup="\${target}.bak.\${stamp}"
  [[ ! -e "\$backup" ]] || backup="\${target}.bak.\${stamp}.\$\$"
  cp -p -- "\$target" "\$backup"; chmod 600 -- "\$backup"
  printf '[torch-dev-env] backed up existing uv config to %s\\n' "\$backup" >&2
fi
umask 077; tmp="\$(mktemp "\${dir}/.uv.toml.tmp.XXXXXX")"; trap 'rm -f -- "\$tmp"' EXIT
printf '%s\\n' "\$template" >"\$tmp"; chmod 600 -- "\$tmp"; mv -f -- "\$tmp" "\$target"
REMOTE_SYNC
}

prompt_sync() {
  local answer
  [[ -t 0 ]] || die 'create requires --sync-uv-config or --no-sync-uv-config when stdin is not a TTY'
  printf "Synchronize the target user's uv source configuration? [y/N] " >&2
  IFS= read -r answer || answer=''
  case "$answer" in [Yy]|[Yy][Ee][Ss]) return 0 ;; *) return 1 ;; esac
}

main() {
  local host='' env_root='' sync='' command='' arg
  local -a args=()
  while (($#)); do
    case "$1" in
      --host) (($# >= 2)) || die '--host requires a value'; host="$2"; shift 2 ;;
      --host=*) host="${1#*=}"; shift ;;
      --env-root) (($# >= 2)) || die '--env-root requires a value'; env_root="$2"; shift 2 ;;
      --env-root=*) env_root="${1#*=}"; shift ;;
      --sync-uv-config) [[ -z "$sync" ]] || die 'sync flags are mutually exclusive'; sync=yes; shift ;;
      --no-sync-uv-config) [[ -z "$sync" ]] || die 'sync flags are mutually exclusive'; sync=no; shift ;;
      --) shift; break ;;
      -*) die "unknown option: $1" ;;
      *) command="$1"; shift; break ;;
    esac
  done
  [[ -n "$env_root" ]] || die '--env-root is required'; [[ -n "$command" ]] || die 'command is required'
  assert_safe_root "$env_root"; assert_safe_host "$host"
  case "$command" in create|install-latest-torch-npu|verify|snapshot-dependencies) ;; *) die "unknown command: $command" ;; esac
  args=("$@")
  if [[ "$command" == create ]]; then
    ((${#args[@]} >= 1)) || die 'create requires one or more selectors'
    for arg in "${args[@]}"; do select_valid "$arg"; done
    if [[ -z "$sync" ]]; then
      show_sync_preview "$host"
      prompt_sync && sync=yes || sync=no
    elif [[ "$sync" == yes ]]; then
      show_sync_preview "$host"
    fi
    [[ "$sync" == yes ]] && { if [[ -n "$host" ]]; then sync_remote "$host"; else sync_local; fi; }
  elif [[ "$command" == snapshot-dependencies ]]; then
    ((${#args[@]} == 1)) || die 'snapshot-dependencies requires OUTPUT_DIR'
    assert_safe_root "${args[0]}" OUTPUT_DIR
    mkdir -p -- "${args[0]}"
  else
    ((${#args[@]} >= 1)) || die "$command requires one or more selectors"
    for arg in "${args[@]}"; do select_valid "$arg"; done
  fi
  [[ -x "$WORKER" ]] || die "worker not found: $WORKER"
  if [[ -n "$host" ]]; then
    if [[ "$command" == snapshot-dependencies ]]; then
      local output="${args[0]}" remote_output stream encoded archive listing name normalized expected
      local -A seen=()
      remote_output="/tmp/torch-dev-env-snapshot-${$}-${RANDOM}"
      stream="$(mktemp "${output}/.a5-remote-snapshot.XXXXXX.stream")"
      encoded="$(mktemp "${output}/.a5-remote-snapshot.XXXXXX.b64")"
      archive="$(mktemp "${output}/.a5-remote-snapshot.XXXXXX.tar")"
      if ! {
        printf '%s\n' 'TORCH_DEV_HOST_MODE=remote' 'exec 3>&1 1>&2'
        cat "$WORKER"
        printf '%s\n' 'exec 1>&3 3>&-' 'remote_output="$3"' 'trap '\''rm -rf -- "$remote_output"'\'' EXIT' 'printf '\''%s\n'\'' TORCH_DEV_SNAPSHOT_BEGIN' 'tar -C "$remote_output" -cf - torch271-py311.freeze.txt torch271-py311.sources.md torch210-py311.freeze.txt torch210-py311.sources.md torch213-py311.freeze.txt torch213-py311.sources.md | base64' 'printf '\''%s\n'\'' TORCH_DEV_SNAPSHOT_END'
      } | "$SSH_BIN" -T "$host" bash --noprofile --norc -s -- "$command" "$env_root" "$remote_output" >"$stream"; then
        rm -f -- "$stream" "$encoded" "$archive"
        return 1
      fi
      if ! awk '
        $0 == "TORCH_DEV_SNAPSHOT_BEGIN" {
          if (begun) exit 3
          begun = 1
          next
        }
        $0 == "TORCH_DEV_SNAPSHOT_END" {
          if (!begun || ended) exit 4
          ended = 1
          exit
        }
        begun { print }
        END {
          if (!begun || !ended) exit 5
        }
      ' "$stream" >"$encoded"; then
        rm -f -- "$stream" "$encoded" "$archive"
        die 'remote snapshot markers are missing or malformed'
      fi
      if ! base64 --decode "$encoded" >"$archive"; then
        rm -f -- "$stream" "$encoded" "$archive"
        die 'remote snapshot payload is not valid base64'
      fi
      rm -f -- "$stream" "$encoded"
      listing="$(tar -tf "$archive")" || { rm -f -- "$archive"; die 'remote snapshot archive is invalid'; }
      while IFS= read -r name; do
        case "$name" in
          ''|./) ;;
          torch271-py311.freeze.txt|torch271-py311.sources.md|torch210-py311.freeze.txt|torch210-py311.sources.md|torch213-py311.freeze.txt|torch213-py311.sources.md) normalized="$name"; seen["$normalized"]=1 ;;
          ./torch271-py311.freeze.txt|./torch271-py311.sources.md|./torch210-py311.freeze.txt|./torch210-py311.sources.md|./torch213-py311.freeze.txt|./torch213-py311.sources.md) normalized="${name#./}"; seen["$normalized"]=1 ;;
          *) rm -f -- "$archive"; die 'remote snapshot archive contains unsafe paths' ;;
        esac
      done <<<"$listing"
      for expected in torch271-py311.freeze.txt torch271-py311.sources.md torch210-py311.freeze.txt torch210-py311.sources.md torch213-py311.freeze.txt torch213-py311.sources.md; do
        [[ -n "${seen[$expected]+present}" ]] || { rm -f -- "$archive"; die "remote snapshot archive is missing $expected"; }
      done
      tar -xf "$archive" -C "$output" --no-same-owner --no-same-permissions
      rm -f -- "$archive"
    else
      { printf '%s\n' 'TORCH_DEV_HOST_MODE=remote'; cat "$WORKER"; } | "$SSH_BIN" -T "$host" bash -s -- "$command" "$env_root" "${args[@]}"
    fi
  else
    TORCH_DEV_HOST_MODE=local bash "$WORKER" "$command" "$env_root" "${args[@]}"
  fi
}

main "$@"
