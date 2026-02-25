"""
Forensic tools for the RepoInvestigator detective.

All git operations run inside a tempfile.TemporaryDirectory() so that
cloned code is never dropped into the live working directory.
subprocess.run() is used instead of os.system() for safety and error capture.
"""

import ast
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Repository Cloning (Sandboxed)
# ---------------------------------------------------------------------------

def clone_repo(repo_url: str, target_dir: str) -> Tuple[bool, str]:
    """
    Clone a GitHub repository into target_dir using subprocess (never os.system).

    Returns:
        (success: bool, message: str)
    """
    # Basic URL sanitisation — reject obviously malicious inputs
    if not repo_url.startswith(("https://github.com/", "https://gitlab.com/")):
        return False, f"Rejected URL (not a known git host): {repo_url}"

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "50", repo_url, target_dir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return False, f"git clone failed: {result.stderr.strip()}"
        return True, "Clone successful"
    except subprocess.TimeoutExpired:
        return False, "git clone timed out after 120 seconds"
    except FileNotFoundError:
        return False, "git executable not found on PATH"
    except Exception as exc:
        return False, f"Unexpected error during clone: {exc}"


# ---------------------------------------------------------------------------
# Git History Extraction
# ---------------------------------------------------------------------------

def extract_git_history(repo_dir: str) -> Dict[str, Any]:
    """
    Run 'git log --oneline --reverse' and return structured commit data.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--reverse", "--format=%h|%ai|%s"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip(), "commits": [], "commit_count": 0}

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        commits = []
        for line in lines:
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({
                    "hash": parts[0],
                    "timestamp": parts[1],
                    "message": parts[2],
                })

        count = len(commits)
        if count <= 1:
            pattern = "bulk_upload"
        elif count <= 3:
            pattern = "minimal"
        else:
            pattern = "atomic"

        return {
            "commits": commits,
            "commit_count": count,
            "pattern": pattern,
            "raw_log": result.stdout.strip(),
        }
    except Exception as exc:
        return {"error": str(exc), "commits": [], "commit_count": 0}


# ---------------------------------------------------------------------------
# File System Scan
# ---------------------------------------------------------------------------

def list_repo_files(repo_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Walk repo_dir and return relative paths of all files.
    Optionally filter by extension list e.g. ['.py', '.md'].
    """
    extensions = extensions or []
    found: List[str] = []
    for root, _dirs, files in os.walk(repo_dir):
        _dirs[:] = [d for d in _dirs if not d.startswith((".", "__"))]
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_dir)
            if not extensions or any(rel.endswith(ext) for ext in extensions):
                found.append(rel)
    return sorted(found)


def read_file_safe(repo_dir: str, rel_path: str) -> Optional[str]:
    """Read a file inside the repo, returning None if it doesn't exist."""
    full = Path(repo_dir) / rel_path
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AST-Based Graph Structure Analysis
# ---------------------------------------------------------------------------

def _collect_names(node: ast.AST) -> List[str]:
    """Recursively collect all Name/Attribute strings from an AST node."""
    names = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.append(child.id)
        elif isinstance(child, ast.Attribute):
            names.append(child.attr)
    return names


def analyze_graph_structure(repo_dir: str) -> Dict[str, Any]:
    """
    Use Python's ast module to inspect graph.py (and state.py) for:
      - StateGraph instantiation
      - add_edge / add_conditional_edges calls (fan-out detection)
      - Pydantic BaseModel / TypedDict usage
      - operator.add / operator.ior reducers
      - tempfile usage in tools
    """
    results: Dict[str, Any] = {
        "state_graph_found": False,
        "parallel_fan_out": False,
        "evidence_aggregator_node": False,
        "pydantic_basemodel": False,
        "typeddict_used": False,
        "operator_reducers": False,
        "tempfile_sandboxing": False,
        "structured_output_enforcement": False,
        "add_edge_calls": [],
        "node_names": [],
        "raw_snippets": {},
        "errors": [],
    }

    targets = {
        "graph": ["src/graph.py", "graph.py"],
        "state": ["src/state.py", "state.py"],
        "tools_repo": ["src/tools/repo_tools.py"],
        "judges": ["src/nodes/judges.py"],
    }

    def _parse_file(rel_path: str):
        src = read_file_safe(repo_dir, rel_path)
        if src is None:
            return None, None
        try:
            return ast.parse(src), src
        except SyntaxError as e:
            results["errors"].append(f"SyntaxError in {rel_path}: {e}")
            return None, src

    # --- graph.py ---
    for candidate in targets["graph"]:
        tree, src = _parse_file(candidate)
        if tree is None:
            continue
        results["raw_snippets"]["graph"] = src[:2000]

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                names = _collect_names(node)
                if "StateGraph" in names:
                    results["state_graph_found"] = True
                if isinstance(node.func, ast.Attribute) and node.func.attr in (
                    "add_edge", "add_conditional_edges"
                ):
                    try:
                        edge_repr = ast.unparse(node)
                    except Exception:
                        edge_repr = str(node)
                    results["add_edge_calls"].append(edge_repr)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results["node_names"].append(node.name)

        if len(results["add_edge_calls"]) >= 4:
            results["parallel_fan_out"] = True
        if "aggregator" in src.lower() or "evidence_aggregator" in src.lower():
            results["evidence_aggregator_node"] = True
        break

    # --- state.py ---
    for candidate in targets["state"]:
        tree, src = _parse_file(candidate)
        if tree is None:
            continue
        results["raw_snippets"]["state"] = src[:2000]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_names = _collect_names(base)
                    if "BaseModel" in base_names:
                        results["pydantic_basemodel"] = True
                    if "TypedDict" in base_names:
                        results["typeddict_used"] = True

        if "operator.add" in src or "operator.ior" in src:
            results["operator_reducers"] = True
        break

    # --- repo_tools.py ---
    for candidate in targets["tools_repo"]:
        tree, src = _parse_file(candidate)
        if tree is None:
            continue
        if "tempfile" in src:
            results["tempfile_sandboxing"] = True
        break

    # --- judges.py ---
    for candidate in targets["judges"]:
        tree, src = _parse_file(candidate)
        if tree is None:
            continue
        if "with_structured_output" in src or "bind_tools" in src:
            results["structured_output_enforcement"] = True
        break

    return results


# ---------------------------------------------------------------------------
# High-Level Entry Point (used by the RepoInvestigator node)
# ---------------------------------------------------------------------------

def full_repo_analysis(repo_url: str) -> Dict[str, Any]:
    """
    Clone the repo into a temp dir, run ALL forensic checks inside the
    context manager (so files still exist during analysis), capture key
    file contents, then clean up.

    The critical fix: analysis must happen INSIDE the try block while the
    temp directory still exists. Using finally: tmp.cleanup() guarantees
    cleanup even if an exception occurs mid-analysis.
    """
    tmp = tempfile.TemporaryDirectory()
    try:
        repo_dir = os.path.join(tmp.name, "repo")
        success, message = clone_repo(repo_url, repo_dir)
        if not success:
            return {"clone_error": message, "repo_url": repo_url}

        # All analysis runs here — temp dir is still alive
        files = list_repo_files(repo_dir, extensions=[".py", ".md", ".toml", ".json"])
        git_history = extract_git_history(repo_dir)
        graph_analysis = analyze_graph_structure(repo_dir)

        # Capture key file contents before temp dir is cleaned up
        key_snippets = {}
        for rel_path in [
            "src/state.py",
            "src/graph.py",
            "src/tools/repo_tools.py",
            "src/nodes/detectives.py",
            "src/nodes/judges.py",
            "src/nodes/justice.py",
        ]:
            content = read_file_safe(repo_dir, rel_path)
            if content:
                key_snippets[rel_path] = content[:3000]

        return {
            "repo_url": repo_url,
            "clone_status": "success",
            "files": files,
            "git_history": git_history,
            "graph_analysis": graph_analysis,
            "key_snippets": key_snippets,
        }
    finally:
        tmp.cleanup()  # Guaranteed cleanup regardless of exceptions
