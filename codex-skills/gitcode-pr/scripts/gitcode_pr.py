#!/usr/bin/env python3
"""Deterministic helpers for repository-aware GitCode issue and PR workflows.

All output is JSON. Commands that can mutate GitCode are previews unless
--execute is supplied. Tokens are accepted only from an environment variable or
an explicitly enabled Git credential helper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "https://api.gitcode.com/api/v5"
DEFAULT_WEB_BASE = "https://gitcode.com"
EXIT_INPUT = 2
EXIT_API = 3
EXIT_CANDIDATE = 4
EXIT_VERIFY = 5


class ToolError(Exception):
    exit_code = EXIT_INPUT


class ApiError(ToolError):
    exit_code = EXIT_API

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def may_be_ambiguous_write(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500


def emit(value: Any, *, stream: Any = None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")


def run_git(repo: Path, *args: str, required: bool = True) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return proc.stdout.rstrip("\r\n")
    if not required:
        return None
    detail = proc.stderr.strip() or proc.stdout.strip() or "unknown git error"
    raise ToolError(f"git {' '.join(args)} failed: {detail}")


def parse_remote_url(url: str) -> dict[str, str]:
    """Parse HTTPS, ssh://, or SCP-style Git remotes without shell heuristics."""
    original = url.strip()
    if not original:
        raise ToolError("remote URL is empty")

    scheme = ""
    host = ""
    path = ""
    if "://" in original:
        parsed = urlsplit(original)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = parsed.path
    else:
        match = re.fullmatch(r"(?:(?P<user>[^@/:]+)@)?(?P<host>[^/:]+):(?P<path>.+)", original)
        if not match:
            raise ToolError(f"unsupported remote URL format: {original}")
        scheme = "scp"
        host = match.group("host")
        path = match.group("path")

    clean_path = path.strip("/")
    if clean_path.endswith(".git"):
        clean_path = clean_path[:-4]
    parts = [part for part in clean_path.split("/") if part]
    if not host or len(parts) < 2:
        raise ToolError(f"remote URL does not contain host/namespace/repository: {original}")

    namespace = "/".join(parts[:-1])
    repository = parts[-1]
    web_scheme = scheme if scheme in {"http", "https"} else "https"
    return {
        "host": host.lower(),
        "namespace": namespace,
        "repository": repository,
        "scheme": scheme,
        "url": original,
        "web_url": f"{web_scheme}://{host}/{namespace}/{repository}",
    }


def _remote(repo: Path, name: str) -> dict[str, Any]:
    fetch_url = run_git(repo, "remote", "get-url", name)
    push_url = run_git(repo, "remote", "get-url", "--push", name)
    if fetch_url is None or push_url is None:
        raise ToolError(f"remote {name!r} did not return fetch and push URLs")
    result: dict[str, Any] = {
        "name": name,
        "fetch_url": fetch_url,
        "push_url": push_url,
    }
    try:
        result["fetch"] = parse_remote_url(fetch_url)
    except ToolError as exc:
        result["fetch_parse_error"] = str(exc)
    try:
        result["push"] = parse_remote_url(push_url)
    except ToolError as exc:
        result["push_parse_error"] = str(exc)
    return result


def _conventions(root: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path]] = []
    for path in root.glob("CONTRIBUTING*"):
        if path.is_file():
            candidates.append(("contribution-guide", path))

    for pattern, kind in (
        (".gitcode/*PULL_REQUEST_TEMPLATE*", "pull-request-template"),
        (".gitcode/*pull_request_template*", "pull-request-template"),
        (".gitcode/ISSUE_TEMPLATE/**/*", "issue-template"),
        (".gitcode/issue_template/**/*", "issue-template"),
        (".github/*PULL_REQUEST_TEMPLATE*", "pull-request-template"),
        (".github/ISSUE_TEMPLATE/**/*", "issue-template"),
    ):
        for path in root.glob(pattern):
            if path.is_file():
                candidates.append((kind, path))

    seen: set[Path] = set()
    result: list[dict[str, Any]] = []
    for kind, path in sorted(candidates, key=lambda item: str(item[1]).lower()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(
            {
                "kind": kind,
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
            }
        )
    return result


def _resolve_commit(repo: Path, refs: Iterable[str]) -> dict[str, str] | None:
    for ref in refs:
        sha = run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}", required=False)
        if sha:
            return {"ref": ref, "sha": sha}
    return None


def command_inspect(args: argparse.Namespace) -> int:
    requested_repo = Path(args.repo).expanduser().resolve()
    root_text = run_git(requested_repo, "rev-parse", "--show-toplevel")
    if root_text is None:
        raise ToolError(f"cannot resolve repository root from {requested_repo}")
    root = Path(root_text).resolve()

    branch = run_git(root, "symbolic-ref", "--quiet", "--short", "HEAD", required=False)
    head_sha = run_git(root, "rev-parse", "HEAD")
    tracking = run_git(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        required=False,
    )
    status_text = run_git(root, "status", "--short") or ""
    status_entries = status_text.splitlines() if status_text else []

    remote_names_text = run_git(root, "remote") or ""
    remote_names = remote_names_text.splitlines() if remote_names_text else []
    remotes = {name: _remote(root, name) for name in remote_names}

    def selected_remote(name: str | None, role: str) -> dict[str, Any] | None:
        if not name:
            return None
        if name not in remotes:
            raise ToolError(f"{role} remote {name!r} does not exist")
        return remotes[name]

    fork = selected_remote(args.fork_remote, "fork")
    upstream = selected_remote(args.upstream_remote, "upstream")
    warnings: list[str] = []
    unresolved: list[str] = []

    expected_host = urlsplit(args.web_base).hostname
    for role, remote in (("fork", fork), ("upstream", upstream)):
        if not remote:
            unresolved.append(f"{role}_remote")
            continue
        parsed = remote.get("push" if role == "fork" else "fetch")
        if not parsed:
            unresolved.append(f"{role}_namespace_repository")
            continue
        parsed_host = str(parsed["host"]).split(":", 1)[0]
        if expected_host and parsed_host.lower() != expected_host.lower():
            warnings.append(
                f"{role} remote host {parsed['host']!r} differs from web host {expected_host!r}"
            )

    if not branch:
        warnings.append("HEAD is detached; supply an explicit publication branch")
        unresolved.append("source_branch")
    if status_entries:
        warnings.append("worktree has uncommitted entries; they are not part of head.sha")

    base_resolution = None
    comparison = None
    if args.base:
        base_refs: list[str] = []
        if args.upstream_remote:
            base_refs.append(f"refs/remotes/{args.upstream_remote}/{args.base}")
        base_refs.extend((args.base, f"refs/heads/{args.base}"))
        base_resolution = _resolve_commit(root, base_refs)
        if not base_resolution:
            warnings.append("target base does not resolve locally; its remote-tracking ref may be missing or stale")
        elif head_sha:
            counts_text = run_git(
                root,
                "rev-list",
                "--left-right",
                "--count",
                f"{base_resolution['ref']}...HEAD",
            )
            changed_text = run_git(
                root,
                "diff",
                "--name-only",
                f"{base_resolution['ref']}...HEAD",
            )
            counts = (counts_text or "0 0").split()
            behind = int(counts[0])
            ahead = int(counts[1])
            changed_files = changed_text.splitlines() if changed_text else []
            comparison = {
                "ahead": ahead,
                "behind": behind,
                "changed_file_count": len(changed_files),
            }
            if ahead == 0:
                warnings.append("source has no commits ahead of the resolved target base")
            if behind:
                warnings.append(
                    f"source is {behind} commit(s) behind the resolved target base"
                )
    else:
        unresolved.append("target_base")

    fork_identity = None
    upstream_identity = None
    if fork and fork.get("push"):
        fork_identity = {
            "namespace": fork["push"]["namespace"],
            "repository": fork["push"]["repository"],
        }
    if upstream and upstream.get("fetch"):
        upstream_identity = {
            "namespace": upstream["fetch"]["namespace"],
            "repository": upstream["fetch"]["repository"],
        }

    relationship = None
    suggested_head = None
    if fork_identity and branch:
        if upstream_identity and fork_identity == upstream_identity:
            relationship = "same-repository"
            suggested_head = branch
        else:
            relationship = "fork-or-cross-repository"
            suggested_head = f"{fork_identity['namespace']}:{branch}"

    result = {
        "mode": "inspect",
        "repository": str(root),
        "head": {"branch": branch, "sha": head_sha, "tracking": tracking},
        "worktree": {"clean": not status_entries, "entries": status_entries},
        "remotes": remotes,
        "fork": fork_identity,
        "upstream": upstream_identity,
        "relationship": relationship,
        "suggested_pr_head": suggested_head,
        "target_base": {
            "branch": args.base,
            "local_resolution": base_resolution,
            "comparison_to_head": comparison,
        },
        "conventions": _conventions(root),
        "unresolved": sorted(set(unresolved)),
        "warnings": warnings,
    }
    emit(result)
    return 0


def _credential_token(host: str) -> str | None:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input=f"protocol=https\nhost={host}\n\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError("git credential fill timed out with prompting disabled") from exc
    if proc.returncode != 0:
        return None
    fields: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    return fields.get("password") or None


def _token(args: argparse.Namespace, *, required: bool) -> str | None:
    token = os.environ.get(args.token_env)
    if not token and args.credential_helper:
        credential_host = args.credential_host or urlsplit(args.web_base).hostname
        if not credential_host:
            raise ToolError(
                "cannot derive credential host from --web-base; supply --credential-host"
            )
        token = _credential_token(credential_host)
    if required and not token:
        raise ApiError(
            f"no API token found in environment variable {args.token_env!r}; "
            "set it or opt into --credential-helper"
        )
    return token


def _error_detail(raw: bytes, token: str | None) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if token:
        text = text.replace(token, "<redacted>")
    if not text:
        return "empty response body"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("message", "error_description", "error"):
                value = parsed.get(key)
                if value:
                    return str(value)[:800]
    except json.JSONDecodeError:
        pass
    return text[:800]


class ApiClient:
    def __init__(
        self,
        *,
        api_base: str,
        token: str | None,
        auth_mode: str,
        timeout: float,
        get_retries: int,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.token = token
        self.auth_mode = auth_mode
        self.timeout = timeout
        self.get_retries = max(0, get_retries)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: Any = None,
    ) -> tuple[Any, Any]:
        method = method.upper()
        query = {key: value for key, value in (params or {}).items() if value is not None}
        headers = {
            "Accept": "application/json",
            "User-Agent": "gitcode-pr-skill/2",
        }
        if self.token:
            if self.auth_mode == "query":
                query["access_token"] = self.token
            elif self.auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.token}"
            elif self.auth_mode == "private-token":
                headers["PRIVATE-TOKEN"] = self.token
            else:
                raise ToolError(f"unsupported authentication mode: {self.auth_mode}")

        endpoint_label = f"{self.api_base}/{path.lstrip('/')}"
        url = endpoint_label
        if query:
            url = f"{url}?{urlencode(query, doseq=True)}"
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        attempts = self.get_retries + 1 if method in {"GET", "HEAD"} else 1
        for attempt in range(attempts):
            request = Request(url, data=data, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    if not raw:
                        return {}, response.headers
                    try:
                        return json.loads(raw), response.headers
                    except json.JSONDecodeError as exc:
                        raise ApiError(
                            f"{method} {endpoint_label} returned non-JSON data: "
                            f"{_error_detail(raw, self.token)}",
                            status=response.status,
                        ) from exc
            except HTTPError as exc:
                raw = exc.read()
                should_retry = method in {"GET", "HEAD"} and (
                    exc.code == 429 or exc.code >= 500
                )
                if should_retry and attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 3))
                    continue
                raise ApiError(
                    f"{method} {endpoint_label} failed with HTTP {exc.code}: "
                    f"{_error_detail(raw, self.token)}",
                    status=exc.code,
                ) from exc
            except URLError as exc:
                if method in {"GET", "HEAD"} and attempt + 1 < attempts:
                    time.sleep(min(2**attempt, 3))
                    continue
                raise ApiError(
                    f"{method} {endpoint_label} failed: {exc.reason}", status=None
                ) from exc
        raise ApiError(f"{method} {endpoint_label} exhausted retries")

    def collection(
        self,
        path: str,
        *,
        params: dict[str, Any] | None,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        previous_signature: str | None = None
        page = 1
        per_page = 100
        while page <= max(1, max_pages):
            page_params = dict(params or {})
            page_params.update({"page": page, "per_page": per_page})
            data, headers = self.request("GET", path, params=page_params)
            items = _collection_items(data)
            signature = json.dumps(items, ensure_ascii=False, sort_keys=True)
            if page > 1 and signature == previous_signature:
                break
            previous_signature = signature
            collected.extend(item for item in items if isinstance(item, dict))

            next_page = str(headers.get("X-Next-Page", "")).strip()
            link = str(headers.get("Link", ""))
            if next_page:
                try:
                    page = int(next_page)
                    continue
                except ValueError:
                    pass
            if 'rel="next"' in link or "rel=next" in link:
                page += 1
                continue
            if len(items) < per_page:
                break
            page += 1
        return collected


def _collection_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "data", "list", "issues", "pulls"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                try:
                    return _collection_items(value)
                except ApiError:
                    pass
    raise ApiError("API collection response has an unsupported JSON shape")


def _client(args: argparse.Namespace, *, token_required: bool) -> ApiClient:
    return ApiClient(
        api_base=args.api_base,
        token=_token(args, required=token_required),
        auth_mode=args.auth_mode,
        timeout=args.timeout,
        get_retries=args.get_retries,
    )


def _component(value: str) -> str:
    return quote(value, safe="")


def _issue_create_path(owner: str) -> str:
    return f"repos/{_component(owner)}/issues"


def _issue_list_path(owner: str, repo: str) -> str:
    return f"repos/{_component(owner)}/{_component(repo)}/issues"


def _pull_collection_path(owner: str, repo: str) -> str:
    return f"repos/{_component(owner)}/{_component(repo)}/pulls"


def _pull_path(owner: str, repo: str, number: str | int) -> str:
    return f"{_pull_collection_path(owner, repo)}/{_component(str(number))}"


def _linked_path(owner: str, repo: str, number: str | int) -> str:
    return f"{_pull_path(owner, repo, number)}/issues"


def _number(item: dict[str, Any]) -> Any:
    return item.get("number", item.get("iid"))


def _web_url(item: dict[str, Any]) -> Any:
    return item.get("html_url") or item.get("web_url") or item.get("url")


def _issue_summary(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": _number(issue),
        "state": issue.get("state"),
        "title": issue.get("title"),
        "url": _web_url(issue),
    }


def _nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _head_owner(head: dict[str, Any]) -> str | None:
    for keys in (
        ("repo", "owner", "login"),
        ("repo", "owner", "name"),
        ("repo", "namespace"),
        ("user", "login"),
        ("user", "name"),
    ):
        value = _nested(head, *keys)
        if isinstance(value, (str, int)) and value:
            return str(value)
        if isinstance(value, dict):
            for key in ("path", "full_path", "login", "name"):
                nested_value = value.get(key)
                if nested_value:
                    return str(nested_value)
    full_path = _nested(head, "repo", "full_path")
    if isinstance(full_path, str) and "/" in full_path:
        return full_path.rsplit("/", 1)[0]
    label = head.get("label")
    if isinstance(label, str) and ":" in label:
        return label.rsplit(":", 1)[0]
    return None


def _pr_summary(pr: dict[str, Any]) -> dict[str, Any]:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    return {
        "number": _number(pr),
        "state": pr.get("state"),
        "title": pr.get("title"),
        "url": _web_url(pr),
        "head": {
            "label": head.get("label"),
            "owner": _head_owner(head),
            "ref": head.get("ref"),
            "sha": head.get("sha"),
        },
        "base": {"ref": base.get("ref"), "sha": base.get("sha")},
    }


def _normalized_title(value: Any) -> str:
    return " ".join(str(value or "").split())


def _find_issues(
    client: ApiClient,
    *,
    owner: str,
    repo: str,
    title: str,
    max_pages: int,
) -> list[dict[str, Any]]:
    issues = client.collection(
        _issue_list_path(owner, repo),
        params={"state": "all", "search": title},
        max_pages=max_pages,
    )
    expected = _normalized_title(title)
    return [issue for issue in issues if _normalized_title(issue.get("title")) == expected]


def _split_head(head: str) -> tuple[str | None, str]:
    if ":" not in head:
        return None, head
    namespace, branch = head.rsplit(":", 1)
    if not namespace or not branch:
        raise ToolError("--head must be <branch> or <namespace>:<branch>")
    return namespace, branch


def _classify_pr(pr: dict[str, Any], expected_head: str, expected_base: str) -> str:
    expected_owner, expected_branch = _split_head(expected_head)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    actual_branch = head.get("ref")
    label = head.get("label")
    if not actual_branch and isinstance(label, str) and ":" in label:
        actual_branch = label.rsplit(":", 1)[1]
    actual_base = base.get("ref")
    if actual_branch and str(actual_branch) != expected_branch:
        return "different"
    if actual_base and str(actual_base) != expected_base:
        return "different"
    if not actual_branch or not actual_base:
        return "ambiguous"
    if expected_owner is None:
        return "exact"

    if isinstance(label, str) and label == expected_head:
        return "exact"
    actual_owner = _head_owner(head)
    if actual_owner is None:
        return "ambiguous"
    return "exact" if actual_owner == expected_owner else "different"


def _find_prs(
    client: ApiClient,
    *,
    owner: str,
    repo: str,
    head: str,
    base: str,
    max_pages: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pulls = client.collection(
        _pull_collection_path(owner, repo),
        params={"state": "open"},
        max_pages=max_pages,
    )
    exact: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for pull in pulls:
        classification = _classify_pr(pull, head, base)
        if classification == "exact":
            exact.append(pull)
        elif classification == "ambiguous":
            ambiguous.append(pull)
    return exact, ambiguous


def command_find_issue(args: argparse.Namespace) -> int:
    client = _client(args, token_required=False)
    matches = _find_issues(
        client,
        owner=args.owner,
        repo=args.repo,
        title=args.title,
        max_pages=args.max_pages,
    )
    emit(
        {
            "operation": "find_issue",
            "match": "exact-normalized-title",
            "candidates": [_issue_summary(issue) for issue in matches],
        }
    )
    return 0


def command_find_pr(args: argparse.Namespace) -> int:
    client = _client(args, token_required=False)
    exact, ambiguous = _find_prs(
        client,
        owner=args.owner,
        repo=args.repo,
        head=args.head,
        base=args.base,
        max_pages=args.max_pages,
    )
    emit(
        {
            "operation": "find_pr",
            "head": args.head,
            "base": args.base,
            "exact": [_pr_summary(pr) for pr in exact],
            "ambiguous": [_pr_summary(pr) for pr in ambiguous],
        }
    )
    return 0


def _read_body(path: str) -> str:
    if path == "-":
        body = sys.stdin.read()
    else:
        try:
            body = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"cannot read body file {path!r}: {exc}") from exc
    if not body.strip():
        raise ToolError("body must not be empty")
    return body


def _issue_url(web_base: str, owner: str, repo: str, number: Any) -> str:
    return f"{web_base.rstrip('/')}/{owner}/{repo}/issues/{number}"


def command_create_issue(args: argparse.Namespace) -> int:
    body = _read_body(args.body_file)
    payload = {"repo": args.repo, "title": args.title, "body": body}
    preview = {
        "operation": "create_issue",
        "execute": bool(args.execute),
        "target": {"owner": args.owner, "repo": args.repo},
        "payload": payload,
    }
    if not args.execute:
        preview["status"] = "previewed"
        preview["preflight_on_execute"] = "exact-title candidate search"
        emit(preview)
        return 0

    client = _client(args, token_required=True)
    candidates = _find_issues(
        client,
        owner=args.owner,
        repo=args.repo,
        title=args.title,
        max_pages=args.max_pages,
    )
    if candidates and not args.allow_duplicate:
        emit(
            {
                "operation": "create_issue",
                "status": "blocked_existing_candidates",
                "candidates": [_issue_summary(issue) for issue in candidates],
            }
        )
        return EXIT_CANDIDATE

    try:
        created, _ = client.request(
            "POST", _issue_create_path(args.owner), payload=payload
        )
    except ApiError as exc:
        if not exc.may_be_ambiguous_write:
            raise
        recovered = _find_issues(
            client,
            owner=args.owner,
            repo=args.repo,
            title=args.title,
            max_pages=args.max_pages,
        )
        if recovered:
            emit(
                {
                    "operation": "create_issue",
                    "status": "recovered_after_ambiguous_error",
                    "warning": str(exc),
                    "candidates": [_issue_summary(issue) for issue in recovered],
                }
            )
            return 0 if len(recovered) == 1 else EXIT_CANDIDATE
        raise

    if not isinstance(created, dict):
        raise ApiError("create-issue response is not a JSON object")
    number = _number(created)
    if number is None:
        raise ApiError("create-issue response did not include number or iid")
    try:
        read_back_candidates = _find_issues(
            client,
            owner=args.owner,
            repo=args.repo,
            title=args.title,
            max_pages=args.max_pages,
        )
    except ApiError as exc:
        emit(
            {
                "operation": "create_issue",
                "status": "created_unverified",
                "created": _issue_summary(created),
                "read_back_error": str(exc),
            }
        )
        return EXIT_VERIFY
    read_back = next(
        (item for item in read_back_candidates if str(_number(item)) == str(number)),
        None,
    )
    if read_back is None:
        emit(
            {
                "operation": "create_issue",
                "status": "created_unverified",
                "created": _issue_summary(created),
                "read_back_candidates": [
                    _issue_summary(item) for item in read_back_candidates
                ],
            }
        )
        return EXIT_VERIFY
    summary = _issue_summary(read_back)
    if not summary["url"]:
        summary["url"] = _issue_url(args.web_base, args.owner, args.repo, number)
    emit(
        {
            "operation": "create_issue",
            "status": "created_and_read_back",
            "issue": summary,
        }
    )
    return 0


def command_create_pr(args: argparse.Namespace) -> int:
    body = _read_body(args.body_file)
    if args.issue_url and args.issue_url not in body:
        raise ToolError("--issue-url was supplied but the exact URL is absent from the PR body")
    payload = {
        "title": args.title,
        "head": args.head,
        "base": args.base,
        "body": body,
    }
    if not args.execute:
        emit(
            {
                "operation": "create_pr",
                "status": "previewed",
                "execute": False,
                "target": {"owner": args.owner, "repo": args.repo},
                "payload": payload,
                "preflight_on_execute": "open head/base PR search",
            }
        )
        return 0

    client = _client(args, token_required=True)
    exact, ambiguous = _find_prs(
        client,
        owner=args.owner,
        repo=args.repo,
        head=args.head,
        base=args.base,
        max_pages=args.max_pages,
    )
    if exact:
        emit(
            {
                "operation": "create_pr",
                "status": "reused_existing",
                "pull_requests": [_pr_summary(pr) for pr in exact],
            }
        )
        return 0 if len(exact) == 1 else EXIT_CANDIDATE
    if ambiguous:
        emit(
            {
                "operation": "create_pr",
                "status": "blocked_ambiguous_candidates",
                "pull_requests": [_pr_summary(pr) for pr in ambiguous],
            }
        )
        return EXIT_CANDIDATE

    try:
        created, _ = client.request(
            "POST", _pull_collection_path(args.owner, args.repo), payload=payload
        )
    except ApiError as exc:
        if not exc.may_be_ambiguous_write:
            raise
        recovered, uncertain = _find_prs(
            client,
            owner=args.owner,
            repo=args.repo,
            head=args.head,
            base=args.base,
            max_pages=args.max_pages,
        )
        if recovered or uncertain:
            emit(
                {
                    "operation": "create_pr",
                    "status": "recovered_after_ambiguous_error",
                    "warning": str(exc),
                    "exact": [_pr_summary(pr) for pr in recovered],
                    "ambiguous": [_pr_summary(pr) for pr in uncertain],
                }
            )
            return 0 if len(recovered) == 1 and not uncertain else EXIT_CANDIDATE
        raise

    if not isinstance(created, dict):
        raise ApiError("create-PR response is not a JSON object")
    number = _number(created)
    if number is None:
        raise ApiError("create-PR response did not include number or iid")
    try:
        read_back, _ = client.request("GET", _pull_path(args.owner, args.repo, number))
    except ApiError as exc:
        emit(
            {
                "operation": "create_pr",
                "status": "created_unverified",
                "created": _pr_summary(created),
                "read_back_error": str(exc),
            }
        )
        return EXIT_VERIFY
    if not isinstance(read_back, dict):
        raise ApiError("PR read-back response is not a JSON object")
    emit(
        {
            "operation": "create_pr",
            "status": "created_and_read_back",
            "pull_request": _pr_summary(read_back),
        }
    )
    return 0


def _linked_numbers(data: Any) -> set[str]:
    return {
        str(_number(item))
        for item in _collection_items(data)
        if isinstance(item, dict) and _number(item) is not None
    }


def command_link_issue(args: argparse.Namespace) -> int:
    client = _client(args, token_required=bool(args.execute))
    path = _linked_path(args.owner, args.repo, args.pr_number)
    linked_data, _ = client.request("GET", path)
    linked = _linked_numbers(linked_data)
    expected = str(args.issue_number)
    if expected in linked:
        emit(
            {
                "operation": "link_issue",
                "status": "already_linked",
                "issue_number": expected,
                "linked_issue_numbers": sorted(linked),
            }
        )
        return 0
    if not args.execute:
        emit(
            {
                "operation": "link_issue",
                "status": "previewed",
                "execute": False,
                "issue_number": expected,
                "currently_linked": sorted(linked),
                "payload": [args.issue_number],
            }
        )
        return 0

    try:
        client.request("POST", path, payload=[args.issue_number])
    except ApiError as exc:
        if exc.status == 403:
            emit(
                {
                    "operation": "link_issue",
                    "status": "permission_denied",
                    "issue_number": expected,
                    "error": str(exc),
                }
            )
            return EXIT_API
        raise
    read_back, _ = client.request("GET", path)
    linked_after = _linked_numbers(read_back)
    ok = expected in linked_after
    emit(
        {
            "operation": "link_issue",
            "status": "linked_and_verified" if ok else "link_not_confirmed",
            "issue_number": expected,
            "linked_issue_numbers": sorted(linked_after),
        }
    )
    return 0 if ok else EXIT_VERIFY


def _actual_head_label(pr: dict[str, Any]) -> str | None:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    label = head.get("label")
    if isinstance(label, str) and ":" in label:
        return str(label)
    owner = _head_owner(head)
    ref = head.get("ref")
    if owner and ref:
        return f"{owner}:{ref}"
    if label:
        return str(label)
    return str(ref) if ref else None


def _head_matches(pr: dict[str, Any], expected: str) -> tuple[bool, str | None]:
    expected_owner, expected_branch = _split_head(expected)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    actual_branch = head.get("ref")
    actual_label = _actual_head_label(pr)
    if actual_branch != expected_branch:
        return False, actual_label
    if expected_owner is None:
        return True, actual_label
    actual_owner = _head_owner(head)
    return actual_owner == expected_owner, actual_label


def command_verify(args: argparse.Namespace) -> int:
    client = _client(args, token_required=False)
    raw_pr, _ = client.request("GET", _pull_path(args.owner, args.repo, args.pr_number))
    if not isinstance(raw_pr, dict):
        raise ApiError("PR read-back response is not a JSON object")
    summary = _pr_summary(raw_pr)
    checks: dict[str, dict[str, Any]] = {}

    expected_states = {state.strip() for state in args.expected_state.split(",") if state.strip()}
    checks["state"] = {
        "ok": str(raw_pr.get("state")) in expected_states,
        "expected": sorted(expected_states),
        "actual": raw_pr.get("state"),
    }
    if args.expected_head:
        matches, actual = _head_matches(raw_pr, args.expected_head)
        checks["head"] = {
            "ok": matches,
            "expected": args.expected_head,
            "actual": actual,
        }
    if args.expected_base:
        actual = _nested(raw_pr, "base", "ref")
        checks["base"] = {
            "ok": actual == args.expected_base,
            "expected": args.expected_base,
            "actual": actual,
        }
    if args.expected_sha:
        actual = _nested(raw_pr, "head", "sha")
        checks["sha"] = {
            "ok": actual == args.expected_sha,
            "expected": args.expected_sha,
            "actual": actual,
        }
    if args.issue_url:
        actual_body = str(raw_pr.get("body") or "")
        checks["issue_url_in_body"] = {
            "ok": args.issue_url in actual_body,
            "expected": args.issue_url,
            "actual": args.issue_url if args.issue_url in actual_body else None,
        }
    if args.issue_number is not None:
        linked_data, _ = client.request(
            "GET", _linked_path(args.owner, args.repo, args.pr_number)
        )
        linked = _linked_numbers(linked_data)
        expected = str(args.issue_number)
        checks["linked_issue"] = {
            "ok": expected in linked,
            "expected": expected,
            "actual": sorted(linked),
        }

    ok = all(check["ok"] for check in checks.values())
    emit(
        {
            "operation": "verify",
            "status": "verified" if ok else "mismatch",
            "ok": ok,
            "pull_request": summary,
            "checks": checks,
        }
    )
    return 0 if ok else EXIT_VERIFY


def _add_api_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--web-base", default=DEFAULT_WEB_BASE)
    parser.add_argument("--token-env", default="GITCODE_TOKEN")
    parser.add_argument(
        "--auth-mode",
        choices=("query", "bearer", "private-token"),
        default="query",
    )
    parser.add_argument("--credential-helper", action="store_true")
    parser.add_argument("--credential-host")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--get-retries", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and safely operate GitCode issue/PR workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect local Git state")
    inspect_parser.add_argument("--repo", default=".")
    inspect_parser.add_argument("--fork-remote")
    inspect_parser.add_argument("--upstream-remote")
    inspect_parser.add_argument("--base")
    inspect_parser.add_argument("--web-base", default=DEFAULT_WEB_BASE)
    inspect_parser.set_defaults(func=command_inspect)

    find_issue = subparsers.add_parser("find-issue", help="find exact-title issue candidates")
    find_issue.add_argument("--owner", required=True)
    find_issue.add_argument("--repo", required=True)
    find_issue.add_argument("--title", required=True)
    _add_api_options(find_issue)
    find_issue.set_defaults(func=command_find_issue)

    find_pr = subparsers.add_parser("find-pr", help="find a PR by head and base")
    find_pr.add_argument("--owner", required=True)
    find_pr.add_argument("--repo", required=True)
    find_pr.add_argument("--head", required=True)
    find_pr.add_argument("--base", required=True)
    _add_api_options(find_pr)
    find_pr.set_defaults(func=command_find_pr)

    create_issue = subparsers.add_parser(
        "create-issue", help="preview or create an issue"
    )
    create_issue.add_argument("--owner", required=True)
    create_issue.add_argument("--repo", required=True)
    create_issue.add_argument("--title", required=True)
    create_issue.add_argument("--body-file", required=True)
    create_issue.add_argument("--execute", action="store_true")
    create_issue.add_argument("--allow-duplicate", action="store_true")
    _add_api_options(create_issue)
    create_issue.set_defaults(func=command_create_issue)

    create_pr = subparsers.add_parser("create-pr", help="preview or create a PR")
    create_pr.add_argument("--owner", required=True)
    create_pr.add_argument("--repo", required=True)
    create_pr.add_argument("--title", required=True)
    create_pr.add_argument("--head", required=True)
    create_pr.add_argument("--base", required=True)
    create_pr.add_argument("--body-file", required=True)
    create_pr.add_argument("--issue-url")
    create_pr.add_argument("--execute", action="store_true")
    _add_api_options(create_pr)
    create_pr.set_defaults(func=command_create_pr)

    link_issue = subparsers.add_parser(
        "link-issue", help="preview or explicitly link an issue to a PR"
    )
    link_issue.add_argument("--owner", required=True)
    link_issue.add_argument("--repo", required=True)
    link_issue.add_argument("--pr-number", required=True)
    link_issue.add_argument("--issue-number", required=True, type=int)
    link_issue.add_argument("--execute", action="store_true")
    _add_api_options(link_issue)
    link_issue.set_defaults(func=command_link_issue)

    verify = subparsers.add_parser("verify", help="read back and verify a PR")
    verify.add_argument("--owner", required=True)
    verify.add_argument("--repo", required=True)
    verify.add_argument("--pr-number", required=True)
    verify.add_argument("--expected-head")
    verify.add_argument("--expected-base")
    verify.add_argument("--expected-sha")
    verify.add_argument("--expected-state", default="open,opened")
    verify.add_argument("--issue-number", type=int)
    verify.add_argument("--issue-url")
    _add_api_options(verify)
    verify.set_defaults(func=command_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ApiError as exc:
        emit(
            {"status": "error", "kind": "api", "message": str(exc)},
            stream=sys.stderr,
        )
        return exc.exit_code
    except ToolError as exc:
        emit(
            {"status": "error", "kind": "input", "message": str(exc)},
            stream=sys.stderr,
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
