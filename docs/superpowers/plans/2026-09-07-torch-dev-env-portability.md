# Torch Dev Environment Skill Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reusable `torch-dev-env` skill with no source-machine content, validate its local/SSH behavior, push it to the `yixuan` GitHub fork, and open a cross-repository GitHub PR against `origin/main`.

**Architecture:** Keep a small skill surface consisting of `SKILL.md`, UI metadata, one public controller, one shared worker, and a public uv source template. Remove captured environment artifacts entirely. Add a repository test that exercises behavior with fake commands and recursively rejects machine identity, absolute source paths, local wheel URLs, timestamps, and the former skill identity.

**Tech Stack:** Bash, uv, SSH, Git, GitHub fork pull requests, Codex skill validator.

---

## File Map

- Modify: `codex-skills/torch-dev-env/SKILL.md` — portable instructions and `torch-dev-env` identity.
- Modify: `codex-skills/torch-dev-env/agents/openai.yaml` — UI identity and `$torch-dev-env` prompt.
- Modify: `codex-skills/torch-dev-env/scripts/manage-torch-env.sh` — neutral configuration variables, log prefixes, and transport.
- Modify: `codex-skills/torch-dev-env/scripts/torch-env-worker.sh` — neutral configuration variables and machine-independent runtime discovery.
- Keep: `codex-skills/torch-dev-env/references/uv-config.toml` — public Huawei Cloud PyPI template.
- Delete: `codex-skills/torch-dev-env/references/environment-manifest.md` — captured host state.
- Delete: `codex-skills/torch-dev-env/references/dependencies/` — captured dependency and source-machine evidence.
- Create: `codex-skills/torch-dev-env/tests/test-torch-dev-env.sh` — behavior and recursive sanitization test.
- Keep: `docs/superpowers/specs/2026-09-07-torch-dev-env-portability-design.md` — approved design.
- Create: `docs/superpowers/plans/2026-09-07-torch-dev-env-portability.md` — this plan.

## Task 1: Establish an isolated feature branch and RED sanitization test

**Files:** Create `codex-skills/torch-dev-env/tests/test-torch-dev-env.sh`.

- [ ] **Step 1: Preserve the synchronized untracked skill and create the feature branch**

The user requires the final files at `~/triton/agent-skills/codex-skills/torch-dev-env`, so work in that checkout rather than moving the skill to another worktree. Confirm the only untracked implementation is the skill directory, then create the branch:

```bash
cd /home/yixuan/triton/agent-skills
git status --short
git switch -c codex/torch-dev-env
```

Expected: current branch becomes `codex/torch-dev-env`; the untracked skill remains in place. Do not stage unrelated paths.

- [ ] **Step 2: Write the recursive failing test**

Create `codex-skills/torch-dev-env/tests/test-torch-dev-env.sh` before changing production files. It must:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

grep -Fx -- 'name: torch-dev-env' "${SKILL_DIR}/SKILL.md" >/dev/null || fail 'frontmatter name is not torch-dev-env'
grep -F -- '$torch-dev-env' "${SKILL_DIR}/agents/openai.yaml" >/dev/null || fail 'default prompt uses the wrong skill name'

forbidden='source-account-id|source-host-alias|a5-torch-env|A5_|/home/|/data/|/mnt/|file://|20[0-9]{2}-[0-9]{2}-[0-9]{2}T|codex_work'
if rg -n -i "${forbidden}" "${SKILL_DIR}" \
  -g '!tests/test-torch-dev-env.sh'; then
  fail 'machine-specific or former identity content remains'
fi

[[ ! -e "${SKILL_DIR}/references/environment-manifest.md" ]] || fail 'captured manifest remains'
[[ ! -e "${SKILL_DIR}/references/dependencies" ]] || fail 'captured dependency snapshots remain'

printf 'PASS\n'
```

During implementation, replace the symbolic `source-account-id` and `source-host-alias` alternatives in the test with exact literal forbidden strings from the approved requirement. Store those literals only in the test script's own exclusion-safe data or construct them from fragments so the recursive scan verifies production files without reintroducing them into distributable instructions.

- [ ] **Step 3: Run RED and record the expected failures**

Run:

```bash
bash codex-skills/torch-dev-env/tests/test-torch-dev-env.sh
```

Expected: failure on the current `a5-torch-env` frontmatter and machine-specific snapshot content. Preserve this output in the task log, not in the skill.

## Task 2: Rename and sanitize skill identity

**Files:** Modify `SKILL.md`, `agents/openai.yaml`, `manage-torch-env.sh`, and `torch-env-worker.sh`.

- [ ] **Step 1: Rename metadata and user-facing identity**

Set the frontmatter to:

```yaml
---
name: torch-dev-env
description: Use when creating, verifying, inspecting, or updating local or SSH-accessible Python 3.11 environments for supported CPU PyTorch and TorchNPU version families.
---
```

Set UI metadata to quoted values:

```yaml
interface:
  display_name: "Torch Development Environments"
  short_description: "Manage portable PyTorch and TorchNPU uv environments"
  default_prompt: "Use $torch-dev-env with a required --env-root and an optional --host to manage a Torch development environment."
```

- [ ] **Step 2: Rename all internal configuration identifiers**

Use these neutral variables consistently:

```text
TORCH_DEV_WORKER
TORCH_DEV_SSH
TORCH_DEV_UV
TORCH_DEV_CURL
TORCH_DEV_SHA256SUM
TORCH_DEV_METADATA_PYTHON
TORCH_DEV_UV_CONFIG_TEMPLATE
TORCH_DEV_HOST_MODE
```

Change log prefixes to `[torch-dev-env]` and `[torch-dev-worker]`. Remove all former aliases rather than retaining backwards compatibility.

- [ ] **Step 3: Remove source-machine paths and assumptions**

Ensure the scripts:

- discover Python 3.11 with uv or the selected environment;
- use guarded temporary directories for verification;
- never change into a fixed project checkout;
- derive uv configuration from runtime `XDG_CONFIG_HOME` or `HOME`;
- require caller-supplied `--env-root`;
- require caller-supplied `--host` only for SSH mode;
- do not contain concrete user, host, cache, project, or interpreter paths.

- [ ] **Step 4: Run the identity portion of the test**

Run the test. Expected: identity checks pass; snapshot-removal checks may still fail until Task 3.

## Task 3: Remove captured evidence and rewrite portable instructions

**Files:** Delete `references/environment-manifest.md` and `references/dependencies/`; modify `SKILL.md`.

- [ ] **Step 1: Remove captured files**

Delete only these confirmed skill resources:

```text
codex-skills/torch-dev-env/references/environment-manifest.md
codex-skills/torch-dev-env/references/dependencies/
```

These files are untracked synchronized copies and contain source-machine evidence; no archived copy is added to the repository.

- [ ] **Step 2: Rewrite SKILL.md around runtime inputs**

Document this interface without concrete host or filesystem examples:

```bash
SKILL_DIR=/path/to/torch-dev-env
bash "${SKILL_DIR}/scripts/manage-torch-env.sh" \
  --host "${SSH_HOST:-}" \
  --env-root "${ENV_ROOT}" \
  --no-sync-uv-config \
  verify 2_10
```

Include:

- local mode when `--host` is omitted or empty;
- SSH mode with a caller-supplied host;
- required absolute `--env-root`;
- selectors `2_7`, `2_10`, and `2_13` and their package matrix;
- create-time uv synchronization decision;
- current-only checksum-verified TorchNPU installation;
- runtime snapshot output to a caller-provided directory;
- no claim that the skill bundles or remembers environment snapshots.

Do not include a default machine, user, home directory, project checkout, cache directory, or captured version timestamp.

- [ ] **Step 3: Run the recursive test and verify GREEN**

Run:

```bash
bash codex-skills/torch-dev-env/tests/test-torch-dev-env.sh
```

Expected: `PASS`.

## Task 4: Restore and extend behavior regression coverage

**Files:** Modify `tests/test-torch-dev-env.sh` only unless a discovered bug requires a tested production fix.

- [ ] **Step 1: Add fake local and SSH fixtures**

Under a guarded `mktemp -d /tmp/torch-dev-env-test-XXXXXX` directory, create fake uv, SSH, downloader, checksum, and Python commands. Ensure cleanup accepts only that prefix.

- [ ] **Step 2: Test controller behavior**

Add assertions for:

- omitted `--host` and `--host ""` both dispatch locally;
- a safe supplied host dispatches over fake SSH;
- missing or relative `--env-root` fails;
- unsafe host and path strings fail before dispatch;
- `create` requires a synchronization decision in non-interactive mode;
- sync flags are mutually exclusive;
- identical config is not rewritten;
- absent config is atomically created with mode `600`;
- differing config is backed up before replacement;
- `--no-sync-uv-config` leaves config unchanged.

- [ ] **Step 3: Test worker behavior**

Add assertions for:

- exact selector mapping and legacy rejection;
- existing destination refusal;
- current PTA URL construction from detected Torch/Python/architecture;
- checksum mismatch and wrong wheel tag rejection;
- `--force-reinstall --no-deps` installation;
- safe verification work directory;
- runtime snapshot creates three `.freeze.txt` and three `.sources.md` files in the caller's output directory;
- SSH snapshot framing ignores banner text and extracts only the six allowed files.

- [ ] **Step 4: Run all tests**

Run:

```bash
bash codex-skills/torch-dev-env/tests/test-torch-dev-env.sh
bash -n codex-skills/torch-dev-env/scripts/manage-torch-env.sh
bash -n codex-skills/torch-dev-env/scripts/torch-env-worker.sh
```

Expected: all commands exit zero and the test prints `PASS`.

## Task 5: Validate, review, and commit the portable skill

**Files:** All retained files under `codex-skills/torch-dev-env`.

- [ ] **Step 1: Validate skill structure**

Run:

```bash
/mnt/c/Users/yixuan1/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  /home/yixuan/triton/agent-skills/codex-skills/torch-dev-env
```

Expected: `Skill is valid!`.

- [ ] **Step 2: Run fresh privacy and repository checks**

Run the recursive test, `git diff --check`, and a manual `rg` scan over the skill. Confirm that remaining absolute paths, if any, are only generic placeholders inside examples and are not real machine locations. Confirm the captured manifest/dependency directory is absent.

- [ ] **Step 3: Review staged scope**

Stage only:

```text
codex-skills/torch-dev-env/
docs/superpowers/specs/2026-09-07-torch-dev-env-portability-design.md
docs/superpowers/plans/2026-09-07-torch-dev-env-portability.md
```

Inspect `git diff --cached --name-status` and `git diff --cached --check`. Do not include unrelated files.

- [ ] **Step 4: Commit**

Run:

```bash
git commit -m "feat: add portable torch development environment skill"
```

Expected: a new commit on `codex/torch-dev-env` containing the sanitized skill and its tests.

## Task 6: Push to the GitHub fork and open the GitHub PR

**Files:** No file changes expected.

- [ ] **Step 1: Verify remotes and authentication without exposing credentials**

Confirm:

```text
origin  https://github.com/HinPeng/agent-skills.git
yixuan  https://github.com/sheny1xuan/agent-skills.git
```

Use `gh auth status` if GitHub CLI is installed. If it is unavailable, use an existing GitHub token from the environment with the REST API; never print the token. If neither authentication method exists, push the branch and report that PR creation requires credentials.

- [ ] **Step 2: Rebase or merge current origin/main only if necessary**

Fetch both remotes and inspect the branch range. Resolve no unrelated conflicts by assumption. Run the complete validation suite again after any integration change.

- [ ] **Step 3: Push the feature branch to yixuan**

Run:

```bash
git push -u yixuan codex/torch-dev-env
```

Expected: `sheny1xuan/agent-skills:codex/torch-dev-env` is updated.

- [ ] **Step 4: Create the cross-repository GitHub PR**

Create a PR with:

```text
base repository: HinPeng/agent-skills
base branch: main
head repository owner: sheny1xuan
head branch: codex/torch-dev-env
title: feat: add portable torch development environment skill
```

Use this body:

```markdown
## Purpose

Adds the `torch-dev-env` Codex skill for creating, verifying, inspecting, and updating uv-based CPU PyTorch and TorchNPU development environments locally or over SSH.

## Usage

- Supply `--env-root` for the environment parent directory.
- Omit `--host` (or pass an empty value) for local execution; supply an SSH host for remote execution.
- Select a supported family with `2_7`, `2_10`, or `2_13`.
- Use `create`, `verify`, `install-latest-torch-npu`, or `snapshot-dependencies`.
- During creation, explicitly choose whether to synchronize the bundled public uv source template.

## Safety and portability

- Refuses unsafe paths and existing destinations.
- Verifies TorchNPU wheel metadata and SHA256 before installation.
- Keeps runtime dependency snapshots outside the distributed skill.
- Contains no machine-specific host, account, filesystem, timestamp, or local-wheel evidence.

## Validation

- Codex skill validator
- Bash syntax checks
- Local/SSH behavior tests using fake tools
- Recursive machine-specific-content scan
```

Use only the GitHub CLI or GitHub REST API and a purpose-built GitHub PR body.

- [ ] **Step 5: Verify and report the PR**

Read the created PR from GitHub and confirm title, base, head, body, and open state. Return the clickable GitHub PR URL and the pushed commit SHA to the user.

## Self-Review Checklist

- [ ] Every approved design requirement maps to a task: identity rename, machine-data removal, generic runtime behavior, privacy testing, fork push, and GitHub PR.
- [ ] The publication workflow uses only GitHub remotes, authentication, APIs, and PR conventions.
- [ ] No captured dependency artifact is committed.
- [ ] Tests precede production edits and include a witnessed RED failure.
- [ ] The final PR body explains both purpose and use.
