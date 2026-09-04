# GitCode API notes

Read this reference only for API-backed discovery, creation, linkage, or verification. Endpoint behavior can vary by GitCode deployment; keep the API and web bases configurable and treat read-back as authoritative.

## Defaults and overrides

The helper defaults to:

- API base: `https://api.gitcode.com/api/v5`
- web base: `https://gitcode.com`
- token environment variable: `GITCODE_TOKEN`
- authentication mode: `query`, for compatibility with the API v5 workflow represented by this skill

Override these with `--api-base`, `--web-base`, `--token-env`, and `--auth-mode`. Supported authentication modes are `query`, `bearer`, and `private-token`; select a non-default mode only when the target deployment supports it.

Tokens are never accepted as command-line values. If the configured environment variable is empty, opt into Git's credential helper with:

```bash
--credential-helper --credential-host <web-host>
```

The helper requests `protocol=https` credentials with terminal prompting disabled and uses the password field as the API token. It never prints the credential response. Do not enable shell tracing around authenticated commands.

Read-only API commands may run anonymously for public repositories. Mutating commands with `--execute` require a token.

## Operations used by the helper

| Operation | Method and path | Important fields |
| --- | --- | --- |
| List repository issues | `GET /repos/{owner}/{repo}/issues` | query `state`, pagination |
| Create issue | `POST /repos/{owner}/issues` | `repo`, `title`, `body` |
| List PRs | `GET /repos/{owner}/{repo}/pulls` | query `state`, pagination; filtered locally by head/base |
| Create PR | `POST /repos/{owner}/{repo}/pulls` | `title`, `head`, `base`, `body` |
| Read PR | `GET /repos/{owner}/{repo}/pulls/{number}` | source, target, SHA, state, URL, body |
| List linked issues | `GET /repos/{owner}/{repo}/pulls/{number}/issues` | issue number |
| Link issue | `POST /repos/{owner}/{repo}/pulls/{number}/issues` | JSON array containing the issue number |

The web UI may call the object a Pull Request or Merge Request. Do not construct the final web URL from that terminology; use the URL returned by the API read-back.

## Authentication and output safety

The Python helper performs requests in-process, so a query-mode token is not interpolated into a `curl` command or exposed as a command-line argument. Errors report a redacted endpoint without its query string.

Even so:

- do not store a token in an issue/PR body or state file;
- do not pass a token through `--body-file` or extra metadata;
- do not paste raw credential-helper output into the conversation;
- prefer a short-lived, least-privilege token supported by the deployment.

## Idempotence and pagination

Collection reads are paginated. The helper accepts both top-level arrays and common wrappers such as `items`, `data`, `list`, `issues`, or `pulls`.

Issue deduplication asks the API to search by title, then applies an exact normalized-title filter locally. It deliberately does not reuse a candidate automatically because two legitimate reports may share a title.

PR deduplication uses source namespace/branch plus target base. When the response omits enough source metadata to prove the namespace, inspect the returned candidates instead of assuming a match.

POST requests are never retried automatically. If a POST fails in a way that may have reached the server, the helper performs one read-after-error lookup. More writes require a new, evidence-based decision.

## Response variations

The helper tolerates these common variations:

- object number in `number` or `iid`;
- web URL in `html_url`, `web_url`, or `url`;
- PR state `open` or `opened`;
- source/base metadata nested under `head` and `base`;
- collections returned directly or under a wrapper key.

If a deployment returns a different shape, preserve the raw response outside user-visible logs only as needed for diagnosis, update the parser narrowly, and do not weaken verification globally.

## Failure handling

- **401/403 authentication:** verify token source and scope. A linkage-specific 403 does not prove the PR-body relationship failed.
- **404:** recheck API base, upstream owner/repository, object number, and token visibility before assuming absence.
- **409/422 conflict:** find an existing issue/PR and inspect validation details; do not change the target silently.
- **429 or 5xx on GET:** the helper retries read-only requests a small bounded number of times.
- **429 or 5xx on POST:** do not repeat the POST; use the recovery lookup and then resume mode.
- **Unexpected non-JSON response:** treat it as an API failure and report status plus a redacted, bounded response excerpt.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | successful read, preview, creation, reuse, or verified no-op |
| `2` | invalid input or local repository error |
| `3` | authentication, network, or API error |
| `4` | creation blocked by an existing candidate requiring a decision |
| `5` | verification completed with one or more mismatches |

Machine-readable JSON goes to standard output. Structured errors go to standard error. Tokens are excluded from both.
