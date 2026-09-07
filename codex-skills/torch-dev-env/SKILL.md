---
name: torch-dev-env
description: Use when creating, updating, verifying, or snapshotting CPU PyTorch and matching TorchNPU uv environments locally or over SSH.
---

# Torch Development Environments

Use the public controller with caller-supplied transport and paths. The same worker runs locally or over SSH, refuses unsafe input and existing destinations, and installs CPU PyTorch before the matching current TorchNPU build.

## Interface

```text
scripts/manage-torch-env.sh [--host SSH_HOST] --env-root ABSOLUTE_PATH \
  [--sync-uv-config | --no-sync-uv-config] COMMAND [ARGUMENTS]
```

- Omit `--host` or pass an empty value for local execution; pass a host for SSH execution.
- Always provide an absolute, shell-safe `--env-root`.
- Select `2_7`, `2_10`, or `2_13`; legacy numeric selectors are rejected.

| Selector | Directory | torch | torchvision | torchaudio |
|---|---|---:|---:|---:|
| `2_7` | `torch271-py311` | 2.7.1+cpu | 0.22.1+cpu | 2.7.1+cpu |
| `2_10` | `torch210-py311` | 2.10.0+cpu | 0.25.0+cpu | 2.10.0+cpu |
| `2_13` | `torch213-py311` | 2.13.0+cpu | 0.28.0+cpu | 2.11.0+cpu |

Commands:

- `create SELECTOR...` creates missing environments, installs the CPU matrix, installs current TorchNPU, and verifies it.
- `install-latest-torch-npu SELECTOR...` updates only TorchNPU in an existing matching environment.
- `verify SELECTOR...` checks imports, versions, a CPU tensor operation, and `uv pip check`.
- `snapshot-dependencies OUTPUT_DIR` writes per-environment freeze and source evidence to the caller's output directory.

## uv source synchronization

During `create`, explicitly choose whether to synchronize the target user's uv config with [references/uv-config.toml](references/uv-config.toml). With neither flag, a TTY shows the target and proposed diff, then prompts `[y/N]`; Enter means no. Non-interactive calls must pass exactly one flag:

```bash
--sync-uv-config
--no-sync-uv-config
```

Synchronization resolves `${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml` at runtime, atomically writes the public template, preserves mode `600`, and backs up a differing file. Declining synchronization leaves it unchanged.

## Examples

Set `SKILL_DIR` to this skill directory. Local:

```bash
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --env-root /srv/torch-envs --no-sync-uv-config create 2_10
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --host '' --env-root /srv/torch-envs verify 2_10
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --env-root /srv/torch-envs snapshot-dependencies /tmp/torch-evidence
```

Remote:

```bash
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --host buildbox --env-root /opt/torch-envs --no-sync-uv-config create 2_10
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --host buildbox --env-root /opt/torch-envs --no-sync-uv-config verify 2_10 2_13
"$SKILL_DIR/scripts/manage-torch-env.sh" \
  --host buildbox --env-root /opt/torch-envs snapshot-dependencies /tmp/torch-evidence
```

Remote snapshots are transferred back to the caller's output directory with framed, validated data so SSH startup text cannot corrupt the archive.

## TorchNPU and snapshots

The worker derives the current public PTA object URL from the installed Torch family, Python ABI, and architecture. It requires one matching wheel and checksum, validates both, installs with `--force-reinstall --no-deps`, and records runtime evidence outside this skill. The public object is current-only; no historical date selector is offered.

Snapshots are generated at runtime and are not bundled with this skill. They include the complete `uv pip freeze`, active uv configuration when available, the managed PyTorch CPU index, PTA URL/checksum evidence, and available `direct_url.json` provenance. Historical indexes that installed ordinary packages cannot be inferred from freeze output.

Run verification from a temporary neutral working directory and use the environment's absolute `bin/python`; an unrelated checkout in the current directory can shadow imports.

Do not delete or replace environments, repair dependencies, or mutate a remote host without explicit authorization.
