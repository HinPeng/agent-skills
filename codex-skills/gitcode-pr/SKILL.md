---
name: gitcode-pr
description: Inspect, prepare, create, resume, and verify GitCode issues and pull requests for local Git repositories, including fork-based and same-repository branches. Use when a GitCode contribution needs repository-aware issue or PR content, branch publication, API execution, or read-back verification. Do not use for other Git forges.
---

# GitCode Issue and PR Workflow

Use repository evidence and explicit user intent to prepare or execute a GitCode handoff. Do not assume a particular owner, repository, remote name, branch, issue type, or contribution template.

## Choose the operating mode

- **Inspect:** discover repository state and conventions without changing local or remote state. Use this by default when the request is ambiguous.
- **Prepare:** produce titles, bodies, resolved parameters, and an operator-visible write plan. Do not push or call mutating APIs.
- **Execute:** perform only the push, issue, PR, or linkage writes the user requested.
- **Resume:** recover from a partial or uncertain run by finding existing objects before attempting any new write.
- **Verify:** read back remote state and report whether the requested outcome exists.

A request to create or push authorizes the named operation; it does not authorize unrelated comments, edits, labels, reviewer changes, force-pushes, or fallback writes.

For prepare, execute, or resume mode, read [references/workflow.md](references/workflow.md). Read [references/gitcode-api.md](references/gitcode-api.md) only when an API call, authentication choice, or API failure is involved.

## Resolve the input contract

Resolve these values from the user's request and repository evidence before any write:

- local repository and source commit SHA;
- source branch and its publication remote;
- upstream GitCode owner/repository and target base branch;
- source namespace for a fork PR, parsed from the remote URL rather than inferred from the remote alias;
- issue mode: `none`, `existing`, or `create`;
- PR mode: `none`, `existing`, or `create`;
- GitCode web/API base URLs and token source when non-default;
- repository contribution instructions, issue templates, and PR template;
- supported claims about the change and its validation.

Do not select a base branch, issue, or remote from an example. If one unresolved value can change the target or create the wrong object, stop before the write and ask for that value. Preparation may continue with the uncertainty clearly marked.

Start discovery with the bundled read-only helper:

```bash
python3 <skill-dir>/scripts/gitcode_pr.py inspect \
  --repo <repository> \
  --fork-remote <fork-remote> \
  --upstream-remote <upstream-remote> \
  --base <target-branch>
```

Omit optional arguments that are genuinely unknown. The JSON output distinguishes discovered facts, unresolved values, and warnings. The remote parser supports HTTPS, `ssh://`, and SCP-style Git URLs.

## Repository conventions take precedence

Before drafting content, inspect the paths returned under `conventions` by the helper. In particular, honor root contribution guides, `.gitcode/PULL_REQUEST_TEMPLATE*`, and GitCode issue templates.

Apply this precedence:

1. explicit instructions in the current user request;
2. repository-local contribution rules and templates;
3. an explicitly selected organization or user profile, if one exists;
4. a minimal generic fallback.

Preserve required headings, checklists, title prefixes, and issue fields. Do not embed one repository's template in the general skill. Do not report tests as passing without concrete evidence.

## Non-negotiable safeguards

- A Git remote alias is not a GitCode namespace. For a fork PR, form `head` from the namespace parsed from the fork URL plus the branch.
- Target the upstream repository in the PR API path and use the resolved upstream branch as `base`.
- When a PR is intended to close or link a cross-repository issue, include the complete issue URL in the PR body. Do not assume `Fixes #N` crosses repositories.
- Treat uncommitted files separately from the committed source SHA. Report them, but do not require a clean worktree merely to push an existing commit.
- Do not add `-u` to a push unless changing branch tracking is requested or clearly intended.
- Do not force-push by default. A non-fast-forward update requires explicit authorization for `--force-with-lease` or an explicitly approved new branch.
- Never print a token, accept it as a command-line value, put it in tracked files, or run credential-bearing commands with shell tracing.
- Before creating an issue or PR, search for a matching existing object. After an ambiguous response, search again instead of retrying the POST.
- After a write, read the object back. Do not infer final source, target, SHA, URL, state, or issue linkage from a create response alone.

## Execute deliberately

The helper's mutating subcommands are previews unless `--execute` is supplied. Use `--execute` only after the values and rendered content have been checked and the requested write is in scope.

Keep the operations independently resumable:

1. publish and verify the source SHA, if a push was requested;
2. resolve or create the issue, if issue mode is not `none`;
3. find or create the PR, if PR mode is not `none`;
4. verify PR source, base, SHA, body, state, and requested issue relationship;
5. attempt explicit issue association only when it is requested or required and read-back has not already confirmed it.

If a step fails, retain and report every confirmed identifier. Resume from read-only discovery; do not repeat completed writes.

## Completion report

Return:

- the source remote, branch, and verified SHA;
- issue number and URL, or `none`;
- PR number and URL, or `none`;
- target repository and base branch;
- read-back checks that passed;
- any unresolved mismatch, permission limitation, or step not requested.

Distinguish `created`, `reused`, `verified`, `previewed`, and `not performed`. A PR-body reference is not proof of explicit API association; report the linked-issues read-back separately.
