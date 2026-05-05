import getpass
import os
import re
from urllib.parse import urlparse

from git import GitCommandError, InvalidGitRepositoryError, Repo
from github import Github, GithubException
from strands import Agent, tool
from strands.models import BedrockModel

# ============================================================
# Configuration — Replace these with your resource IDs
# ============================================================

MODEL_ID = "us.anthropic.claude-opus-4-7"
REGION = "us-east-1"

# Base directory where repositories will be cloned.
# Override with the CODER_WORKSPACE env var if desired.
DEFAULT_WORKSPACE = os.path.expanduser("~/coder-workspace")

# GitHub authentication is read from the GITHUB_TOKEN env var.
# A token with `repo` scope is required to push and open pull requests.
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"


# ============================================================
# Helpers
# ============================================================

def _workspace() -> str:
    path = os.environ.get("CODER_WORKSPACE", DEFAULT_WORKSPACE)
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_repo_path(repo_path: str) -> str:
    """Resolve a repo path relative to the workspace if it isn't absolute."""
    if os.path.isabs(repo_path):
        return repo_path
    return os.path.join(_workspace(), repo_path)


def _infer_repo_name(repo_url: str) -> str:
    """Infer a sensible folder name from a Git URL."""
    cleaned = repo_url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]

    # Handle scp-like SSH syntax: git@github.com:owner/repo
    if "@" in cleaned and "://" not in cleaned and ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    else:
        parsed = urlparse(cleaned)
        if parsed.path:
            cleaned = parsed.path

    name = cleaned.rstrip("/").split("/")[-1]
    return name or "repository"


def _extract_github_slug(remote_url: str) -> str:
    """Extract 'owner/repo' from a GitHub HTTPS or SSH URL."""
    url = remote_url.strip()
    if url.endswith(".git"):
        url = url[: -len(".git")]

    # SSH form: git@github.com:owner/repo
    ssh_match = re.match(r"git@github\.com:(?P<slug>[^/]+/[^/]+)$", url)
    if ssh_match:
        return ssh_match.group("slug")

    # HTTPS form: https://github.com/owner/repo
    parsed = urlparse(url)
    if parsed.netloc.endswith("github.com"):
        return parsed.path.lstrip("/")

    raise ValueError(f"Not a recognizable GitHub URL: {remote_url}")


def _authed_remote_url(remote_url: str, token: str) -> str:
    """Embed a GitHub token into an HTTPS remote URL for push auth."""
    parsed = urlparse(remote_url)
    if parsed.scheme not in ("http", "https"):
        # SSH or other — return unchanged; auth handled by the SSH agent.
        return remote_url
    netloc = parsed.netloc.split("@", 1)[-1]
    return f"{parsed.scheme}://x-access-token:{token}@{netloc}{parsed.path}"


# ============================================================
# Custom Tools
# ============================================================

def _normalize_remote_url(url: str) -> str:
    """Normalize a Git remote URL for comparison (case, .git suffix, scheme)."""
    if not url:
        return ""
    cleaned = url.strip().lower()
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    cleaned = cleaned.rstrip("/")

    # Convert scp-like SSH (git@host:owner/repo) to a comparable form.
    if "@" in cleaned and "://" not in cleaned and ":" in cleaned:
        host, path = cleaned.split(":", 1)
        host = host.split("@", 1)[-1]
        return f"{host}/{path}"

    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.netloc}{parsed.path}"
    return cleaned


@tool
def clone_repository(
    repo_url: str,
    destination: str = "",
    branch: str = "",
    depth: int = 0,
) -> str:
    """Clone a Git repository into the local workspace using GitPython.

    Idempotent: if the target directory already contains a clone of the SAME
    repository (matching `origin` URL), the existing path is reused instead
    of cloning again. If the directory exists with a different repo (or is
    not a git repo at all), the call fails so nothing is overwritten.

    Args:
        repo_url: The Git URL of the repository (HTTPS or SSH).
        destination: Optional folder name (relative to the workspace).
            If empty, the name is inferred from the URL.
        branch: Optional branch or tag to check out after cloning.
        depth: Optional shallow-clone depth. 0 (default) means a full clone.

    Returns:
        A status message including the local path on success.
    """
    if not repo_url:
        return "Error: repo_url is required."

    folder_name = destination.strip() or _infer_repo_name(repo_url)
    target_path = os.path.join(_workspace(), folder_name)
    requested = _normalize_remote_url(repo_url)

    if os.path.exists(target_path):
        try:
            existing_repo = Repo(target_path)
        except InvalidGitRepositoryError:
            return (
                f"Error: '{target_path}' already exists but is not a git repository. "
                f"Remove it or pick a different destination."
            )
        except Exception as exc:
            return f"Error: could not inspect '{target_path}' ({exc})."

        try:
            existing_url = next(existing_repo.remote("origin").urls)
        except (ValueError, StopIteration):
            existing_url = ""

        if _normalize_remote_url(existing_url) == requested:
            return (
                f"Already cloned: {repo_url} is present at {target_path}. "
                f"Skipping clone."
            )

        return (
            f"Error: '{target_path}' already contains a different repository "
            f"(origin={existing_url!r}). Pick a different destination."
        )

    kwargs = {}
    if branch:
        kwargs["branch"] = branch
    if depth and depth > 0:
        kwargs["depth"] = depth

    try:
        Repo.clone_from(repo_url, target_path, **kwargs)
    except GitCommandError as exc:
        return f"Error: git clone failed.\n{exc.stderr or exc}"

    return f"Successfully cloned {repo_url} into {target_path}"


@tool
def create_branch(repo_path: str, branch_name: str, base: str = "") -> str:
    """Create and check out a new branch in a local repository.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        branch_name: Name of the new branch to create.
        base: Optional base branch/commit to branch off from.
            Defaults to the current HEAD.

    Returns:
        A status message indicating which branch is now checked out.
    """
    if not repo_path or not branch_name:
        return "Error: repo_path and branch_name are required."

    full_path = _resolve_repo_path(repo_path)
    if not os.path.isdir(full_path):
        return f"Error: '{full_path}' is not a directory."

    try:
        repo = Repo(full_path)
        if base:
            new_branch = repo.create_head(branch_name, base)
        else:
            new_branch = repo.create_head(branch_name)
        new_branch.checkout()
    except GitCommandError as exc:
        return f"Error: failed to create branch.\n{exc.stderr or exc}"
    except Exception as exc:
        return f"Error: {exc}"

    return f"Created and checked out branch '{branch_name}' in {full_path}"


def _resolve_file_in_repo(repo_path: str, relative_file_path: str) -> str:
    """Resolve a file path inside a repo, guarding against traversal."""
    full_repo = os.path.normpath(_resolve_repo_path(repo_path))
    target = os.path.normpath(os.path.join(full_repo, relative_file_path))
    if target != full_repo and not target.startswith(full_repo + os.sep):
        raise ValueError("relative_file_path escapes the repository directory.")
    return target


def _read_file_slice(target: str, start_line: int, end_line: int) -> str:
    """Read a [start, end] line range from a file with a numbered prefix."""
    try:
        with open(target, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"Error: failed to read file ({exc})."

    total = len(lines)
    start = max(1, start_line)
    end = total if end_line <= 0 else min(end_line, total)
    if start > total:
        return f"File has {total} lines; start_line {start_line} is past EOF."

    width = len(str(end))
    body = "".join(
        f"{str(i).rjust(width)}|{lines[i - 1]}" for i in range(start, end + 1)
    )
    if not body.endswith("\n"):
        body += "\n"
    return f"{target} (lines {start}-{end} of {total}):\n{body}"


def _search_repo(
    repo_path: str,
    query: str,
    case_sensitive: bool,
    max_results: int,
) -> str:
    """Search a repository for `query` using `git grep` (respects .gitignore)."""
    full_path = _resolve_repo_path(repo_path)
    try:
        repo = Repo(full_path)
    except InvalidGitRepositoryError:
        return f"Error: '{full_path}' is not a git repository."
    except Exception as exc:
        return f"Error: could not open repo ({exc})."

    args = ["-n", "-I", "--fixed-strings"]
    if not case_sensitive:
        args.append("-i")
    args.extend(["--", query])

    try:
        raw = repo.git.grep(*args)
    except GitCommandError as exc:
        # `git grep` exits 1 when there are no matches — that's not an error.
        if exc.status == 1 and not (exc.stderr or "").strip():
            return f"No matches for {query!r} in {full_path}."
        return f"Error: git grep failed.\n{exc.stderr or exc}"

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return f"No matches for {query!r} in {full_path}."

    truncated = len(lines) > max_results
    shown = lines[:max_results]

    files: dict[str, list[str]] = {}
    for ln in shown:
        # git grep -n format: "path:lineno:content"
        parts = ln.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, content = parts
        files.setdefault(path, []).append(f"  {lineno.rjust(6)}: {content}")

    blocks = [f"{path}\n" + "\n".join(hits) for path, hits in files.items()]
    header = (
        f"Found {len(lines)} match(es) for {query!r} in {len(files)} file(s)"
        f" — showing first {len(shown)}.\n" if truncated
        else f"Found {len(lines)} match(es) for {query!r} in {len(files)} file(s).\n"
    )
    return header + "\n\n".join(blocks)


@tool
def read_file(
    repo_path: str,
    relative_file_path: str = "",
    query: str = "",
    start_line: int = 1,
    end_line: int = 0,
    case_sensitive: bool = False,
    max_results: int = 30,
) -> str:
    """Read a file OR search a repository for matching text.

    Two mutually exclusive modes:

    1. Read mode (provide `relative_file_path`): returns the file (or a
       line range) prefixed with line numbers. Use this before edit_file
       to copy the exact text you want to replace.

    2. Search mode (provide `query`, leave `relative_file_path` empty):
       runs `git grep` across the repository and returns matching files
       with line numbers and snippets. Respects `.gitignore`. Use this
       when you don't yet know which file contains the symbol/text you
       want to change.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        relative_file_path: File path inside the repo (e.g. "src/app.py").
            Required for read mode.
        query: Text to search for across the repository.
            Required for search mode. Treated as a literal string, not regex.
        start_line: (read mode) 1-indexed first line to return.
        end_line: (read mode) 1-indexed last line; 0 means EOF.
        case_sensitive: (search mode) match exact case. Defaults to false.
        max_results: (search mode) cap on returned matches. Defaults to 30.

    Returns:
        The file slice (read mode) or a summary of matches (search mode).
    """
    if not repo_path:
        return "Error: repo_path is required."

    has_path = bool(relative_file_path.strip())
    has_query = bool(query.strip())
    if has_path == has_query:
        return (
            "Error: provide exactly one of `relative_file_path` (to read a "
            "file) or `query` (to search the repo)."
        )

    if has_path:
        try:
            target = _resolve_file_in_repo(repo_path, relative_file_path)
        except ValueError as exc:
            return f"Error: {exc}"
        if not os.path.isfile(target):
            return f"Error: file not found: {target}"
        return _read_file_slice(target, start_line, end_line)

    return _search_repo(
        repo_path=repo_path,
        query=query,
        case_sensitive=case_sensitive,
        max_results=max(1, max_results),
    )


@tool
def edit_file(
    repo_path: str,
    relative_file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a file by replacing an exact substring with new content.

    This is the primary tool for changing code. Workflow:
      1. Call read_file first to see the current contents.
      2. Choose an old_string that appears exactly once (include enough
         surrounding context to make it unique), unless you intentionally
         set replace_all=True.
      3. Provide the replacement new_string.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        relative_file_path: File path inside the repo (e.g. "src/app.py").
        old_string: Exact text to find. Must match byte-for-byte (including
            whitespace and indentation).
        new_string: Text to insert in place of old_string.
        replace_all: If true, replace every occurrence. If false (default),
            old_string must appear exactly once.

    Returns:
        A status message describing how many occurrences were replaced.
    """
    if not repo_path or not relative_file_path:
        return "Error: repo_path and relative_file_path are required."
    if old_string == "":
        return "Error: old_string must not be empty (use create_file for new files)."
    if old_string == new_string:
        return "Error: old_string and new_string are identical — nothing to change."

    try:
        target = _resolve_file_in_repo(repo_path, relative_file_path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not os.path.isfile(target):
        return f"Error: file not found: {target}"

    try:
        with open(target, "r", encoding="utf-8") as fh:
            original = fh.read()
    except OSError as exc:
        return f"Error: failed to read file ({exc})."

    occurrences = original.count(old_string)
    if occurrences == 0:
        return (
            "Error: old_string was not found in the file. "
            "Re-read the file and copy the exact text (including whitespace)."
        )
    if occurrences > 1 and not replace_all:
        return (
            f"Error: old_string matches {occurrences} places. "
            f"Add more context to make it unique, or set replace_all=true."
        )

    if replace_all:
        updated = original.replace(old_string, new_string)
        replaced = occurrences
    else:
        updated = original.replace(old_string, new_string, 1)
        replaced = 1

    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(updated)
    except OSError as exc:
        return f"Error: failed to write file ({exc})."

    return f"Replaced {replaced} occurrence(s) in {target}."


@tool
def create_file(repo_path: str, relative_file_path: str, content: str) -> str:
    """Create a brand-new file inside a local repository.

    Fails if the file already exists — use edit_file to modify existing files.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        relative_file_path: File path inside the repo (e.g. "src/app.py").
        content: Full contents for the new file.

    Returns:
        A status message with the resolved file path.
    """
    if not repo_path or not relative_file_path:
        return "Error: repo_path and relative_file_path are required."

    try:
        target = _resolve_file_in_repo(repo_path, relative_file_path)
    except ValueError as exc:
        return f"Error: {exc}"

    if os.path.exists(target):
        return f"Error: '{target}' already exists. Use edit_file to modify it."

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        return f"Error: failed to write file ({exc})."

    return f"Created {target} ({len(content)} bytes)."


@tool
def commit_changes(repo_path: str, message: str, add_all: bool = True) -> str:
    """Stage and commit changes in a local repository.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        message: Commit message.
        add_all: If true (default), stage all modified and new files first.

    Returns:
        A status message including the new commit SHA.
    """
    if not repo_path or not message:
        return "Error: repo_path and message are required."

    full_path = _resolve_repo_path(repo_path)
    try:
        repo = Repo(full_path)
        if add_all:
            repo.git.add(A=True)

        if not repo.is_dirty(untracked_files=True) and not repo.index.diff("HEAD"):
            return "Nothing to commit — working tree is clean."

        commit = repo.index.commit(message)
    except GitCommandError as exc:
        return f"Error: failed to commit.\n{exc.stderr or exc}"
    except Exception as exc:
        return f"Error: {exc}"

    return f"Committed {commit.hexsha[:8]} on branch '{repo.active_branch.name}'."


@tool
def push_branch(repo_path: str, remote: str = "origin", branch: str = "") -> str:
    """Push a branch to a remote, authenticating with GITHUB_TOKEN if set.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        remote: Remote name (default "origin").
        branch: Branch to push. Defaults to the currently checked-out branch.

    Returns:
        A status message describing what was pushed.
    """
    if not repo_path:
        return "Error: repo_path is required."

    full_path = _resolve_repo_path(repo_path)
    try:
        repo = Repo(full_path)
        branch_name = branch or repo.active_branch.name
        remote_obj = repo.remote(remote)

        token = os.environ.get(GITHUB_TOKEN_ENV)
        original_url = next(remote_obj.urls)
        push_url = _authed_remote_url(original_url, token) if token else original_url

        # Temporarily swap the URL for the push so the token isn't persisted.
        if push_url != original_url:
            remote_obj.set_url(push_url, original_url)
        try:
            results = remote_obj.push(refspec=f"{branch_name}:{branch_name}", set_upstream=True)
        finally:
            if push_url != original_url:
                remote_obj.set_url(original_url, push_url)

        for r in results:
            # bit 1024 == ERROR per GitPython's PushInfo flags
            if r.flags & 1024:
                return f"Error: push failed for {branch_name}: {r.summary.strip()}"
    except GitCommandError as exc:
        return f"Error: failed to push.\n{exc.stderr or exc}"
    except Exception as exc:
        return f"Error: {exc}"

    return f"Pushed branch '{branch_name}' to remote '{remote}'."


@tool
def open_pull_request(
    repo_path: str,
    title: str,
    body: str = "",
    base: str = "main",
    head: str = "",
    draft: bool = False,
) -> str:
    """Open a GitHub pull request for the given repository.

    Requires the GITHUB_TOKEN environment variable with `repo` scope.

    Args:
        repo_path: Path to the local repo (absolute or relative to workspace).
        title: Pull request title.
        body: Pull request description (Markdown).
        base: Target branch on the upstream repo (default "main").
        head: Source branch. Defaults to the currently checked-out branch.
        draft: If true, open the PR as a draft.

    Returns:
        A status message including the PR URL on success.
    """
    if not repo_path or not title:
        return "Error: repo_path and title are required."

    token = os.environ.get(GITHUB_TOKEN_ENV)
    if not token:
        return f"Error: {GITHUB_TOKEN_ENV} env var is not set."

    full_path = _resolve_repo_path(repo_path)
    try:
        repo = Repo(full_path)
        head_branch = head or repo.active_branch.name
        remote_url = next(repo.remote("origin").urls)
        slug = _extract_github_slug(remote_url)
    except Exception as exc:
        return f"Error: could not resolve GitHub repo from local clone ({exc})."

    try:
        gh = Github(token)
        gh_repo = gh.get_repo(slug)
        pr = gh_repo.create_pull(
            title=title,
            body=body,
            base=base,
            head=head_branch,
            draft=draft,
        )
    except GithubException as exc:
        return f"Error: GitHub API rejected the PR ({exc.status}): {exc.data}"
    except Exception as exc:
        return f"Error: {exc}"

    return f"Opened PR #{pr.number}: {pr.html_url}"


# ============================================================
# Build the Agent
# ============================================================

def create_coder_agent():
    """Create the Coder agent that downloads repos and opens pull requests."""

    os.environ["AWS_REGION"] = REGION

    bedrock_model = BedrockModel(
        model_id=MODEL_ID,
        region_name=REGION,
        max_tokens=2000,
    )

    system_prompt = """You are the Coder agent.
You help developers download Git repositories, change code, push the changes,
and open pull requests on GitHub.

Tools available to you:
- clone_repository: clone a remote repo into the local workspace.
- create_branch: create and check out a new branch for your changes.
- read_file: read a specific file (relative_file_path) OR search the repo
  for text (query) when you don't yet know which file to open.
- edit_file: replace an exact substring (old_string) with new_string.
- create_file: create a brand-new file (fails if it already exists).
- commit_changes: stage and commit pending changes.
- push_branch: push a branch to its remote (uses GITHUB_TOKEN for HTTPS auth).
- open_pull_request: open a GitHub pull request from the pushed branch.

Editing workflow (mandatory):
1. If you don't already know which file to change, call read_file in SEARCH
   mode (pass `query`) to locate candidate files first.
2. Then call read_file in READ mode (pass `relative_file_path`) on the
   chosen file so you have the exact current text — never edit blind.
3. For edit_file, pick an old_string that appears exactly once — include
   enough surrounding context (a few lines above and below) to make it
   unique. Match whitespace and indentation byte-for-byte.
4. If a change spans multiple non-contiguous regions, make multiple
   edit_file calls — one per region.
5. Use create_file ONLY for files that do not exist yet.
6. Never paste a whole-file rewrite into edit_file; keep replacements
   focused on the smallest meaningful chunk.

General workflow guidelines:
- Always work on a new branch (never commit directly to main/master).
- Use small, focused commits with clear messages.
- After pushing, open a pull request with a meaningful title and a body that
  summarises the change and how to test it.
- If a tool returns an error, surface it to the user and stop — do not
  guess or retry blindly.
- Never invent repository URLs, branch names, or file paths."""

    agent = Agent(
        model=bedrock_model,
        tools=[
            clone_repository,
            create_branch,
            read_file,
            edit_file,
            create_file,
            commit_changes,
            push_branch,
            open_pull_request,
        ],
        system_prompt=system_prompt,
    )

    return agent


# ============================================================
# Run the Agent
# ============================================================

def _prompt_github_token() -> None:
    """Prompt for a GitHub token and store it in the environment.

    The input is read via getpass so it is not echoed to the terminal.
    Pressing Enter at the prompt skips token setup; in that case push
    and open_pull_request will return an explicit error if invoked.
    """
    existing = os.environ.get(GITHUB_TOKEN_ENV)
    if existing:
        print(f"Using {GITHUB_TOKEN_ENV} from environment.")
        return

    try:
        token = getpass.getpass(
            "GitHub token (press Enter to skip; needed for push / PRs): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        token = ""

    if token:
        os.environ[GITHUB_TOKEN_ENV] = token
        print("GitHub token configured for this session.")
    else:
        print("No token provided — clone and local edits will work, but "
              "push and open_pull_request will fail until a token is set.")


def main():
    print("Coder Agent — Clone, Edit, Push, and Open Pull Requests")
    print("=" * 60)
    print("Examples:")
    print("  'clone https://github.com/octocat/Hello-World'")
    print("  'in Hello-World, create branch fix/typo, change README.md ...,")
    print("   commit, push, and open a PR against main'")
    print("\nType 'quit' to exit.\n")

    _prompt_github_token()
    print()

    agent = create_coder_agent()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = agent(user_input)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
