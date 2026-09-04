# Workflow details

Use this reference for prepare, execute, and resume modes. It describes decision points; repository-local instructions still take precedence for content and validation requirements.

## 1. Discover facts before choosing actions

Run:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py inspect \
  --repo <repository> \
  --fork-remote <fork-remote> \
  --upstream-remote <upstream-remote> \
  --base <target-branch>
```

The only required argument is `--repo`. Supply remote names and `--base` when known; omitted values remain unresolved rather than being guessed.

Review:

- `head.sha`, `head.branch`, and `head.tracking`;
- `worktree.entries`, which are not part of the pushed commit unless committed;
- parsed `fork` and `upstream` namespace/repository values;
- `suggested_pr_head`;
- resolved base ref, SHA, and ahead/behind comparison with `HEAD`;
- `conventions` paths;
- `unresolved` and `warnings`.

Read relevant contribution guides and templates in full before drafting. Do not rely on a copied template from a previous repository or an older branch.

### Resolving source and target

- Prefer an explicitly named source branch and base branch.
- Otherwise, the checked-out branch is a source candidate, not automatically the target base.
- Treat the branch's tracking configuration as evidence, not as permission to change it.
- Resolve the base against the intended upstream remote when available, for example `upstream/<base>`; note when the remote-tracking ref may be stale.
- For a fork PR, derive `<namespace>:<branch>` from the fork URL. For a same-repository PR, use the API form supported by that repository and verify it on read-back.

If the fork and upstream resolve to different repository names and are not a recognized fork relationship, do not assume GitCode will accept the short `namespace:branch` head form.

## 2. Select the operation matrix

Choose each dimension independently:

| Dimension | Values | Meaning |
| --- | --- | --- |
| Mode | `inspect`, `prepare`, `execute`, `resume`, `verify` | Maximum allowed effect for this run |
| Source | `fork`, `same-repository`, `already-published` | How the PR head is supplied |
| Issue | `none`, `existing`, `create` | Whether an issue is linked or created |
| PR | `none`, `existing`, `create` | Whether a PR is only verified/reused or may be created |

Do not create an issue merely because a PR is requested. Do not create a PR merely because an issue is requested. When the repository requires an issue, surface that requirement during preparation.

## 3. Draft repository-compliant content

### Issue

Choose content by issue type and repository instructions. A generic fallback is:

- problem or objective;
- relevant environment or scope;
- reproduction for a defect, or motivation and design for a feature;
- expected behavior or acceptance criteria;
- evidence and current validation status.

Use only sections that fit the issue. Do not invent a root cause, benchmark, affected version, assignee, label, or test result.

### Pull request

Start from the repository's PR template when present. Preserve required headings and checklists, filling non-applicable required sections explicitly according to repository convention.

Summarize the committed diff rather than uncommitted worktree changes. Include:

- why the change is needed;
- what the committed change does;
- compatibility, documentation, and interface impact when required;
- exact validation commands and observed results;
- limitations or unrun checks.

When linking a cross-repository issue, include its full URL in the form required by the repository. A generic closing reference is:

```text
Fixes https://gitcode.com/<owner>/<repo>/issues/<number>
```

## 4. Preview every intended write

Before execute mode, show the resolved target and content:

- source remote, branch, and local SHA;
- destination owner/repository and base;
- exact push refspec, if any;
- issue action and full title/body;
- PR action, `head`, `base`, and full title/body;
- optional linkage action;
- unknowns and claims not backed by evidence.

The helper also previews API writes when `--execute` is omitted. Body content is read from a file or standard input so it does not need fragile shell quoting.

## 5. Publish the branch only when requested

Check whether the remote ref exists:

```bash
git ls-remote --heads <fork-remote> refs/heads/<branch>
```

Publish without changing local tracking configuration by default:

```bash
git push <fork-remote> HEAD:refs/heads/<branch>
```

Afterward, compare `git rev-parse HEAD` with the SHA returned by `git ls-remote`. If the push is rejected as non-fast-forward, stop. Use `--force-with-lease` only with explicit authorization and only after resolving the expected remote SHA.

## 6. Find before creating

Read [gitcode-api.md](gitcode-api.md) before using API subcommands.

Find an exact-title issue candidate:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py find-issue \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --title <issue-title>
```

Find an existing PR for a resolved head/base pair:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py find-pr \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --head <source-namespace>:<source-branch> \
  --base <target-branch>
```

Exact-title issues are candidates, not automatically the same report. Inspect them before deciding whether to reuse one. An exact head/base PR match is normally resumable; verify its SHA and content before reuse.

## 7. Create with an explicit execution gate

Preview issue creation:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py create-issue \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --title <issue-title> \
  --body-file <issue-body-file>
```

Repeat with `--execute` only when creation is requested. The command searches for an exact-title candidate before POST and blocks rather than choosing for the user. `--allow-duplicate` is reserved for an explicitly reviewed collision.

Preview PR creation:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py create-pr \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --title <pr-title> \
  --head <source-namespace>:<source-branch> \
  --base <target-branch> \
  --body-file <pr-body-file> \
  --issue-url <full-issue-url>
```

Repeat with `--execute` only when creation is requested. The command reuses an existing open PR with the same head/base instead of posting a duplicate.

Do not blindly retry a create request after a timeout or server error. The helper performs one read-after-error recovery search; if the result remains uncertain, switch to resume mode.

## 8. Link only when needed

A full issue URL in the PR body may be sufficient for the intended relationship. When explicit association is requested or repository policy requires it, use the idempotent linkage command:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py link-issue \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --pr-number <pr-number> \
  --issue-number <issue-number>
```

Without `--execute`, this reports whether the issue is already linked and previews a POST only when absent. With `--execute`, it posts once and reads the linked-issues endpoint again. A permission failure does not authorize comments or PR edits as fallbacks.

## 9. Verify or resume

Verify all known invariants:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py verify \
  --owner <upstream-owner> \
  --repo <upstream-repo> \
  --pr-number <pr-number> \
  --expected-head <source-namespace>:<source-branch> \
  --expected-base <target-branch> \
  --expected-sha <source-sha> \
  --issue-number <issue-number> \
  --issue-url <full-issue-url>
```

Omit checks that do not apply. A failed check is a mismatch to report, not permission to edit remote state.

Resume order:

1. locate issue candidates and a head/base PR;
2. GET known objects and retain confirmed numbers and URLs;
3. verify the published source SHA;
4. perform only the still-missing operation that the user requested;
5. read back again.

Report a partial result precisely. For example, distinguish a PR whose body references an issue from one whose linked-issues endpoint confirms association.
