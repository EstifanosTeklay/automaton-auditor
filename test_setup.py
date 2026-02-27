"""
Quick setup test — run this BEFORE the full auditor to verify
everything is installed and configured correctly.

Usage: python test_setup.py
"""

import os
import sys


def check(label, fn):
    try:
        fn()
        print(f"  ✅ {label}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False


print("\n=== Automaton Auditor — Setup Check ===\n")

# 1. Environment
print("1. Environment variables:")
check("ANTHROPIC_API_KEY is set",
      lambda: (_ for _ in ()).throw(Exception("Not set")) if not os.getenv("ANTHROPIC_API_KEY") else None)

# 2. Imports
print("\n2. Package imports:")
check("anthropic",          lambda: __import__("anthropic"))
check("langchain_anthropic",lambda: __import__("langchain_anthropic"))
check("langchain_core",     lambda: __import__("langchain_core"))
check("langgraph",          lambda: __import__("langgraph"))
check("pydantic",           lambda: __import__("pydantic"))
check("dotenv",             lambda: __import__("dotenv"))
check("pdfplumber",         lambda: __import__("pdfplumber"))

# 3. Project imports
print("\n3. Project module imports:")
sys.path.insert(0, os.path.dirname(__file__))
check("src.state",              lambda: __import__("src.state"))
check("src.tools.repo_tools",   lambda: __import__("src.tools.repo_tools"))
check("src.tools.doc_tools",    lambda: __import__("src.tools.doc_tools"))
check("src.nodes.detectives",   lambda: __import__("src.nodes.detectives"))
check("src.nodes.judges",       lambda: __import__("src.nodes.judges"))
check("src.nodes.justice",      lambda: __import__("src.nodes.justice"))
check("src.graph",              lambda: __import__("src.graph"))

# 4. Rubric
print("\n4. Rubric file:")
check("rubric.json exists and loads", lambda: __import__("json").load(open("rubric.json")))

# 5. Git
print("\n5. Git availability:")
import subprocess
check("git is on PATH",
      lambda: subprocess.run(["git", "--version"], capture_output=True, check=True))

# 6. Quick Claude API test
print("\n6. Claude API connectivity:")
def _test_claude():
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "say ok"}],
    )
    assert resp.content[0].text
check("Claude API responds", _test_claude)

print("\n=== Done ===")
print("If all green: run the auditor with:")
print("  python -m src.graph <repo_url> reports/interim_report.pdf\n")
