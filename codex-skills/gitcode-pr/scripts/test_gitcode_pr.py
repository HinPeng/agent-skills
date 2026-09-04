#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout


SCRIPT = Path(__file__).with_name("gitcode_pr.py")
SPEC = importlib.util.spec_from_file_location("gitcode_pr", SCRIPT)
assert SPEC and SPEC.loader
gitcode_pr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gitcode_pr)


class RemoteParsingTests(unittest.TestCase):
    def test_https_remote(self) -> None:
        parsed = gitcode_pr.parse_remote_url("https://gitcode.com/example/project.git")
        self.assertEqual(parsed["host"], "gitcode.com")
        self.assertEqual(parsed["namespace"], "example")
        self.assertEqual(parsed["repository"], "project")

    def test_scp_remote(self) -> None:
        parsed = gitcode_pr.parse_remote_url("git@gitcode.com:example/project.git")
        self.assertEqual(parsed["scheme"], "scp")
        self.assertEqual(parsed["namespace"], "example")
        self.assertEqual(parsed["repository"], "project")

    def test_ssh_remote_and_nested_namespace(self) -> None:
        parsed = gitcode_pr.parse_remote_url(
            "ssh://git@gitcode.example/group/team/project.git"
        )
        self.assertEqual(parsed["namespace"], "group/team")
        self.assertEqual(parsed["repository"], "project")


class ResponseParsingTests(unittest.TestCase):
    def test_collection_wrappers(self) -> None:
        values = [{"number": 1}]
        for response in (values, {"items": values}, {"data": {"list": values}}):
            self.assertEqual(gitcode_pr._collection_items(response), values)

    def test_pr_match_requires_namespace_when_requested(self) -> None:
        exact = {
            "head": {"ref": "topic", "label": "contributor:topic"},
            "base": {"ref": "main"},
        }
        ambiguous = {"head": {"ref": "topic"}, "base": {"ref": "main"}}
        different = {
            "head": {"ref": "topic", "label": "someone-else:topic"},
            "base": {"ref": "main"},
        }
        self.assertEqual(
            gitcode_pr._classify_pr(exact, "contributor:topic", "main"), "exact"
        )
        self.assertEqual(
            gitcode_pr._classify_pr(ambiguous, "contributor:topic", "main"),
            "ambiguous",
        )
        self.assertEqual(
            gitcode_pr._classify_pr(different, "contributor:topic", "main"),
            "different",
        )

    def test_single_pr_namespace_object_is_supported(self) -> None:
        pull = {
            "head": {
                "ref": "topic",
                "label": "topic",
                "repo": {"namespace": {"path": "contributor"}},
            },
            "base": {"ref": "main"},
        }
        self.assertEqual(gitcode_pr._head_owner(pull["head"]), "contributor")
        self.assertEqual(
            gitcode_pr._actual_head_label(pull), "contributor:topic"
        )
        self.assertEqual(
            gitcode_pr._classify_pr(pull, "contributor:topic", "main"), "exact"
        )


class FakeClient:
    def __init__(self, collection_result=None, request_result=None) -> None:
        self.collection_result = collection_result or []
        self.request_result = request_result or []
        self.requests = []

    def collection(self, path, *, params, max_pages):
        self.requests.append(("COLLECTION", path, params))
        return self.collection_result

    def request(self, method, path, *, params=None, payload=None):
        self.requests.append((method, path, payload))
        return self.request_result, {}


class IdempotenceTests(unittest.TestCase):
    def test_create_issue_blocks_existing_candidate_before_post(self) -> None:
        fake = FakeClient(
            collection_result=[
                {
                    "number": 7,
                    "title": "Existing issue",
                    "state": "open",
                    "html_url": "https://gitcode.example/org/repo/issues/7",
                }
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as body_file:
            body_file.write("Issue body\n")
            body_file.flush()
            args = argparse_namespace(
                owner="org",
                repo="repo",
                title=" Existing   issue ",
                body_file=body_file.name,
                execute=True,
                allow_duplicate=False,
                max_pages=5,
                web_base="https://gitcode.example",
            )
            output = io.StringIO()
            with mock.patch.object(gitcode_pr, "_client", return_value=fake):
                with redirect_stdout(output):
                    status = gitcode_pr.command_create_issue(args)
        self.assertEqual(status, gitcode_pr.EXIT_CANDIDATE)
        self.assertEqual(json.loads(output.getvalue())["status"], "blocked_existing_candidates")
        self.assertFalse(any(call[0] == "POST" for call in fake.requests))

    def test_create_pr_reuses_exact_head_base_before_post(self) -> None:
        fake = FakeClient(
            collection_result=[
                {
                    "number": 9,
                    "title": "Existing PR",
                    "state": "open",
                    "head": {
                        "ref": "topic",
                        "label": "contributor:topic",
                        "sha": "abc",
                    },
                    "base": {"ref": "main"},
                }
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as body_file:
            body_file.write("PR body\n")
            body_file.flush()
            args = argparse_namespace(
                owner="org",
                repo="repo",
                title="New PR title",
                head="contributor:topic",
                base="main",
                body_file=body_file.name,
                issue_url=None,
                execute=True,
                max_pages=5,
            )
            output = io.StringIO()
            with mock.patch.object(gitcode_pr, "_client", return_value=fake):
                with redirect_stdout(output):
                    status = gitcode_pr.command_create_pr(args)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "reused_existing")
        self.assertFalse(any(call[0] == "POST" for call in fake.requests))

    def test_link_issue_is_noop_when_read_back_already_contains_it(self) -> None:
        fake = FakeClient(request_result=[{"number": 12}])
        args = argparse_namespace(
            owner="org",
            repo="repo",
            pr_number="4",
            issue_number=12,
            execute=True,
        )
        output = io.StringIO()
        with mock.patch.object(gitcode_pr, "_client", return_value=fake):
            with redirect_stdout(output):
                status = gitcode_pr.command_link_issue(args)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "already_linked")
        self.assertFalse(any(call[0] == "POST" for call in fake.requests))


def argparse_namespace(**values):
    class Namespace:
        pass

    namespace = Namespace()
    for key, value in values.items():
        setattr(namespace, key, value)
    return namespace


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def git(self, repo: Path, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_inspect_discovers_ssh_remote_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.git(repo, "init")
            self.git(repo, "config", "user.name", "Skill Test")
            self.git(repo, "config", "user.email", "skill@example.invalid")
            (repo / "README.md").write_text("fixture\n", encoding="utf-8")
            template = repo / ".gitcode" / "PULL_REQUEST_TEMPLATE.md"
            template.parent.mkdir()
            template.write_text("# Summary\n", encoding="utf-8")
            self.git(repo, "add", "README.md", ".gitcode/PULL_REQUEST_TEMPLATE.md")
            self.git(repo, "commit", "-m", "test fixture")
            self.git(repo, "checkout", "-b", "topic")
            self.git(
                repo,
                "remote",
                "add",
                "origin",
                "git@gitcode.com:contributor/project.git",
            )
            self.git(
                repo,
                "remote",
                "add",
                "upstream",
                "https://gitcode.com/organization/project.git",
            )

            result = self.run_cli(
                "inspect",
                "--repo",
                str(repo),
                "--fork-remote",
                "origin",
                "--upstream-remote",
                "upstream",
                "--base",
                "main",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["suggested_pr_head"], "contributor:topic")
            self.assertEqual(payload["fork"]["namespace"], "contributor")
            convention_paths = {item["path"] for item in payload["conventions"]}
            self.assertIn(".gitcode/PULL_REQUEST_TEMPLATE.md", convention_paths)

    def test_create_previews_do_not_need_network_or_token(self) -> None:
        issue = self.run_cli(
            "create-issue",
            "--owner",
            "organization",
            "--repo",
            "project",
            "--title",
            "Example issue",
            "--body-file",
            "-",
            input_text="Issue body\n",
        )
        self.assertEqual(issue.returncode, 0, issue.stderr)
        self.assertEqual(json.loads(issue.stdout)["status"], "previewed")

        issue_url = "https://gitcode.com/organization/project/issues/12"
        pull = self.run_cli(
            "create-pr",
            "--owner",
            "organization",
            "--repo",
            "project",
            "--title",
            "Example PR",
            "--head",
            "contributor:topic",
            "--base",
            "main",
            "--body-file",
            "-",
            "--issue-url",
            issue_url,
            input_text=f"Summary\n\nFixes {issue_url}\n",
        )
        self.assertEqual(pull.returncode, 0, pull.stderr)
        payload = json.loads(pull.stdout)
        self.assertEqual(payload["status"], "previewed")
        self.assertEqual(payload["payload"]["head"], "contributor:topic")

    def test_issue_url_guard_rejects_mismatched_body(self) -> None:
        result = self.run_cli(
            "create-pr",
            "--owner",
            "organization",
            "--repo",
            "project",
            "--title",
            "Example PR",
            "--head",
            "topic",
            "--base",
            "main",
            "--body-file",
            "-",
            "--issue-url",
            "https://gitcode.com/organization/project/issues/12",
            input_text="No issue reference here\n",
        )
        self.assertEqual(result.returncode, gitcode_pr.EXIT_INPUT)
        self.assertEqual(json.loads(result.stderr)["kind"], "input")


if __name__ == "__main__":
    unittest.main()
