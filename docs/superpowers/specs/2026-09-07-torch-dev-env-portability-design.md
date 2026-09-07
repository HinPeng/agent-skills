# Torch Dev Environment Skill Portability Design

## Goal

Convert `codex-skills/torch-dev-env` into a reusable, machine-independent skill named `torch-dev-env`. Remove all captured machine identity, host-specific state, filesystem-specific evidence, and local artifact provenance while preserving the generic local-or-SSH environment management workflow.

The finished skill must contain no references to the source machine, source account, source filesystem, capture timestamps, local wheel locations, or its former A5-specific identity.

## Scope

Keep these portable capabilities:

- local execution when `--host` is omitted or empty;
- SSH execution when a caller supplies `--host`;
- a caller-supplied absolute `--env-root`;
- selectors `2_7`, `2_10`, and `2_13` mapped to stable environment directory names and compatible Torch packages;
- optional synchronization of a public uv source template;
- creation and verification of CPU PyTorch environments;
- installation of the current matching TorchNPU package from the public Huawei PTA object endpoint;
- dependency snapshots written only to a caller-supplied output directory at runtime.

Remove these non-portable contents:

- captured dependency and source-evidence files;
- the captured environment manifest;
- concrete SSH host names and account names;
- source-machine absolute paths and interpreter locations;
- capture dates, evidence-directory suffixes, and source-machine runtime descriptions;
- `file://` references to locally stored wheels;
- former skill names, A5-specific log prefixes, environment variables, and prose.

Public service URLs and generic technology names are not machine-local data and remain allowed, including the PyTorch CPU index, the Huawei Cloud PyPI mirror, and the Huawei PTA object endpoint.

## Skill Identity

Use `torch-dev-env` consistently in:

- `SKILL.md` frontmatter and title;
- `agents/openai.yaml` display name, summary, and `$torch-dev-env` default prompt;
- shell log prefixes;
- test names and messages;
- documentation examples.

Replace all `A5_*` configuration variables with neutral `TORCH_DEV_*` names. Do not provide compatibility aliases for the former names because that would preserve the forbidden machine-specific identity.

## Public Interface

Keep one public controller:

```text
scripts/manage-torch-env.sh \
  [--host HOST] \
  --env-root ABSOLUTE_PATH \
  [--sync-uv-config | --no-sync-uv-config] \
  COMMAND [ARGUMENTS]
```

The caller supplies every execution-specific value. The skill must not suggest a default host, environment root, account directory, cache root, or project checkout.

Commands remain:

- `create SELECTOR...`
- `install-latest-torch-npu SELECTOR...`
- `verify SELECTOR...`
- `snapshot-dependencies OUTPUT_DIR`

Selectors remain:

| Selector | Environment directory | Torch | TorchVision | TorchAudio |
|---|---|---:|---:|---:|
| `2_7` | `torch271-py311` | 2.7.1 | 0.22.1 | 2.7.1 |
| `2_10` | `torch210-py311` | 2.10.0 | 0.25.0 | 2.10.0 |
| `2_13` | `torch213-py311` | 2.13.0 | 0.28.0 | 2.11.0 |

Reject the ambiguous legacy selector forms.

## Portable Runtime Behavior

### Execution transport

When `--host` is empty, invoke the worker on the current system. When it is non-empty, stream the same worker through SSH. Validate the supplied host and paths before dispatch. Never derive a host or filesystem location from bundled evidence.

### Python and working directories

Discover Python 3.11 through uv or the selected environment. Do not require a source-machine interpreter path or change into a particular project checkout. For verification, create and use a temporary neutral working directory so an unrelated checkout cannot shadow Python packages.

### uv configuration

Keep the public Huawei Cloud PyPI template in `references/uv-config.toml`. Resolve the target user's uv configuration at runtime from `XDG_CONFIG_HOME` or `HOME`; do not mention a captured user configuration path.

Creation still requires an explicit synchronization decision. Interactive execution displays the runtime target and proposed change, then defaults to no. Non-interactive execution requires `--sync-uv-config` or `--no-sync-uv-config`. Back up a differing configuration before atomic replacement.

### TorchNPU

Construct the public PTA object URL from the environment's installed Torch version, Python major/minor, and normalized architecture. Download only the current object, verify the accompanying SHA256, validate wheel tags and metadata, and install with `--force-reinstall --no-deps`.

Record evidence under the caller-supplied environment root at runtime. Do not bundle runtime evidence or locally resolved wheel paths in the skill.

### Dependency snapshots

Keep `snapshot-dependencies`, but write generated `.freeze.txt` and `.sources.md` files only to the caller-provided output directory. Snapshot output describes the runtime target using values supplied or discovered during that invocation. The repository must not ship captured snapshots.

## File Changes

Keep and sanitize:

```text
codex-skills/torch-dev-env/
├── SKILL.md
├── agents/openai.yaml
├── references/uv-config.toml
└── scripts/
    ├── manage-torch-env.sh
    └── torch-env-worker.sh
```

Delete:

```text
codex-skills/torch-dev-env/references/environment-manifest.md
codex-skills/torch-dev-env/references/dependencies/
```

Do not add a README, migration document, archived snapshot, or compatibility wrapper that preserves forbidden details.

## Tests

Add a repository-local test for behavior and sanitization. Follow RED-GREEN-REFACTOR:

1. Run a baseline scan and demonstrate that the synchronized skill currently fails due to its old name and machine-specific content.
2. Update tests before production files.
3. Rename identity and sanitize implementation.
4. Run the same scan and behavior tests until they pass.

The sanitization test must recursively inspect all files inside `codex-skills/torch-dev-env` and fail on:

- the source account identifier;
- the source host alias;
- former skill names or A5-prefixed internal identifiers;
- `/home/`, `/data/`, and `/mnt/` absolute paths;
- `file://` URLs;
- captured timestamps and evidence-directory identifiers;
- bundled dependency snapshot or environment-manifest files.

Behavior tests must verify:

- skill frontmatter name and default prompt use `torch-dev-env`;
- local dispatch when host is omitted or empty;
- remote dispatch with a caller-supplied host;
- mandatory caller-supplied `--env-root`;
- allowed selectors and rejection of legacy forms;
- uv synchronization confirmation and non-interactive flags;
- current PTA URL construction, checksum enforcement, and safe snapshot transfer;
- no hard-coded execution host, user directory, environment root, or project checkout.

Run `quick_validate.py`, `bash -n`, the behavior test, the recursive privacy scan, and `git diff --check` before completion.

## Safety and Compatibility

- Do not mutate any real virtual environment during tests.
- Use fake uv, SSH, downloader, and wheel fixtures under guarded temporary directories.
- Preserve the public command interface except for intentionally renamed internal configuration variables and forbidden legacy identity.
- Refuse existing environment destinations and unsafe caller input as before.
- Keep runtime evidence outside the skill repository.

## Success Criteria

- The skill is named `torch-dev-env` everywhere.
- No file under the skill contains source-machine identity, paths, timestamps, or local artifact references.
- No captured dependency or manifest files remain bundled.
- Local and SSH workflows still share one worker and accept only caller-supplied execution targets.
- uv synchronization, CPU Torch creation, current TorchNPU installation, verification, and runtime snapshots remain tested.
- Skill validation, shell tests, privacy scans, and diff checks pass.
