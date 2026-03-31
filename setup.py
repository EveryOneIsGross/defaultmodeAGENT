#!/usr/bin/env python3
"""defaultMODE Agent — Setup & Environment Configurator

Usage:
    python setup.py            # full setup (venv + install + .env config)
    python setup.py --env      # only configure .env keys
    python setup.py --install  # only create venv and install packages
    python setup.py --help     # show this message
"""

import getpass
import os
import subprocess
import sys
import platform
import argparse
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
VENV_DIR   = ROOT / ".venv"
ENV_FILE   = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
REQUIREMENTS = ROOT / "requirements.txt"
PROMPTS_DIR  = ROOT / "agent" / "prompts"

# Directories inside agent/prompts/ that are not real bots
_SKIP_DIRS = {"archive", "default", "__pycache__"}

# ─────────────────────────────────────────────
# Colours (no deps, just ANSI)
# ─────────────────────────────────────────────
_WIN = sys.platform == "win32"
if _WIN:
    # Enable VT100 on Windows 10+
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

_PINK      = "\033[38;2;255;194;194m"   # #FFC2C2
_PINK_DIM  = "\033[38;2;200;140;140m"   # darker for dimmed context

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = _PINK_DIM
    RED    = _PINK
    GREEN  = _PINK
    YELLOW = _PINK
    CYAN   = _PINK
    WHITE  = _PINK

def _c(color: str, text: str) -> str:
    return f"{color}{text}{C.RESET}"

def ok(msg):    print(_c(C.GREEN,  f"  ✓  {msg}"))
def info(msg):  print(_c(C.CYAN,   f"  →  {msg}"))
def warn(msg):  print(_c(C.YELLOW, f"  ⚠  {msg}"))
def err(msg):   print(_c(C.RED,    f"  ✗  {msg}"))
def hdr(msg):   print(f"\n{_c(C.BOLD + C.WHITE, msg)}")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _python_ok() -> bool:
    v = sys.version_info
    if v < (3, 10):
        err(f"Python 3.10+ required (found {v.major}.{v.minor})")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def _venv_pip() -> Path:
    """Return path to pip inside the venv."""
    if _WIN:
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def _venv_python() -> Path:
    if _WIN:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _activate_hint() -> str:
    if _WIN:
        return str(VENV_DIR / "Scripts" / "activate")
    return f"source {VENV_DIR / 'bin' / 'activate'}"


# ─────────────────────────────────────────────
# Step 1 — virtual environment
# ─────────────────────────────────────────────
def setup_venv() -> bool:
    hdr("Virtual Environment")

    if VENV_DIR.exists():
        ok(f"Venv already exists at {VENV_DIR}")
        return True

    info(f"Creating venv at {VENV_DIR} …")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err("Failed to create venv:")
        print(result.stderr)
        return False

    ok("Venv created")
    return True


# ─────────────────────────────────────────────
# Step 2 — install requirements
# ─────────────────────────────────────────────
def install_requirements() -> bool:
    hdr("Dependencies")

    if not REQUIREMENTS.exists():
        warn("requirements.txt not found — skipping install")
        return True

    pip = _venv_pip()
    if not pip.exists():
        err("pip not found in venv — did venv creation succeed?")
        return False

    info("Upgrading pip …")
    subprocess.run(
        [str(pip), "install", "--quiet", "--upgrade", "pip"],
        check=False
    )

    info("Installing requirements.txt (this may take a minute) …")
    result = subprocess.run(
        [str(pip), "install", "--quiet", "-r", str(REQUIREMENTS)],
        capture_output=False  # let output stream so user can see progress
    )
    if result.returncode != 0:
        err("Some packages failed to install — check output above")
        return False

    ok("All packages installed")
    return True


# ─────────────────────────────────────────────
# Step 3 — .env configuration
# ─────────────────────────────────────────────
def _load_env(path: Path) -> dict[str, str]:
    """Parse a .env file into {key: value}, preserving comments."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" in stripped:
            key, _, val = stripped.partition("=")
            # Strip inline comments
            val = val.split(" #")[0].strip()
            env[key.strip()] = val.strip()
    return env


def _write_env(path: Path, values: dict[str, str]) -> None:
    """
    Merge new values into the .env file.  Existing keys are updated in-place;
    new keys are appended at the end.
    """
    lines: list[str] = []
    updated: set[str] = set()

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=")[0].strip()
                if key in values:
                    lines.append(f"{key}={values[key]}")
                    updated.add(key)
                    continue
            lines.append(line)

    # Append any keys that weren't already in the file
    new_keys = [k for k in values if k not in updated]
    if new_keys:
        lines.append("")
        lines.append("# Added by setup.py")
        for k in new_keys:
            lines.append(f"{k}={values[k]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prompt(label: str, secret: bool = False, default: str = "") -> str:
    prompt_str = f"     {_c(C.CYAN, label)}"
    if default:
        prompt_str += _c(C.DIM, f" [{default}]")
    prompt_str += ": "

    if secret:
        val = getpass.getpass(prompt_str)
    else:
        val = input(prompt_str)

    return val.strip() or default


def _detect_bots() -> list[str]:
    """Return uppercase bot names from agent/prompts/ subdirectories."""
    if not PROMPTS_DIR.exists():
        return []
    return [
        d.name.upper()
        for d in sorted(PROMPTS_DIR.iterdir())
        if d.is_dir() and d.name.lower() not in _SKIP_DIRS
    ]


def configure_env() -> bool:
    hdr(".env Configuration")

    # Bootstrap from example if .env doesn't exist
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            import shutil
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
            ok(f"Created .env from .env.example")
        else:
            ENV_FILE.touch()
            ok("Created empty .env")

    current = _load_env(ENV_FILE)
    updates: dict[str, str] = {}

    def _need(key: str) -> bool:
        """True if the key is absent or empty."""
        return not current.get(key, "").strip()

    # ── Service API Keys ──────────────────────────────────────────────
    hdr("  Service API Keys")
    print(_c(C.DIM, "  Press Enter to skip optional keys.\n"))

    service_keys = [
        ("OPENAI_API_KEY",     "OpenAI API key (sk-…)",          True,  False),
        ("ANTHROPIC_API_KEY",  "Anthropic API key (sk-ant-…)",    True,  False),
        ("GEMINI_API_KEY",     "Google Gemini API key",           True,  False),
        ("OPENROUTER_API_KEY", "OpenRouter API key (sk-or-…)",    True,  False),
        ("VLLM_API_KEY",       "vLLM API key",                    True,  False),
        ("OLLAMA_API_BASE",    "Ollama base URL",                 False, True),
    ]

    for key, label, secret, has_default in service_keys:
        if _need(key):
            default = "http://localhost:11434" if key == "OLLAMA_API_BASE" else ""
            val = _prompt(label, secret=secret, default=default)
            if val:
                updates[key] = val
                ok(f"{key} set")
            else:
                info(f"{key} skipped")
        else:
            ok(f"{key} already configured")

    # ── Per-bot Discord Tokens ────────────────────────────────────────
    bots = _detect_bots()
    if bots:
        hdr("  Bot Tokens")
        print(_c(C.DIM, f"  Detected bots: {', '.join(bots)}\n"))

        for bot in bots:
            discord_key = f"DISCORD_TOKEN_{bot}"
            github_key  = f"GITHUB_TOKEN_{bot}"
            repo_key    = f"GITHUB_REPO_{bot}"

            needs_any = _need(discord_key) or _need(github_key) or _need(repo_key)
            if not needs_any and not any(k in updates for k in (discord_key, github_key, repo_key)):
                ok(f"{bot}: all tokens configured")
                continue

            print(f"\n  {_c(C.BOLD, bot)}")

            if _need(discord_key):
                val = _prompt(f"Discord token", secret=True)
                if val:
                    updates[discord_key] = val
                    ok(f"{discord_key} set")
                else:
                    info(f"{discord_key} skipped")
            else:
                ok(f"{discord_key} already set")

            if _need(github_key):
                val = _prompt(f"GitHub token (optional)", secret=True)
                if val:
                    updates[github_key] = val
                    ok(f"{github_key} set")
            else:
                ok(f"{github_key} already set")

            if _need(repo_key):
                val = _prompt(f"GitHub repo (e.g. user/repo, optional)", secret=False)
                if val:
                    updates[repo_key] = val
                    ok(f"{repo_key} set")
            else:
                ok(f"{repo_key} already set")

    # ── Generic fallback token ────────────────────────────────────────
    if _need("DISCORD_TOKEN"):
        hdr("  Generic Fallback Token")
        val = _prompt("DISCORD_TOKEN (fallback, optional)", secret=True)
        if val:
            updates["DISCORD_TOKEN"] = val
            ok("DISCORD_TOKEN set")

    # ── Write ─────────────────────────────────────────────────────────
    if updates:
        _write_env(ENV_FILE, updates)
        ok(f"Wrote {len(updates)} key(s) to .env")
    else:
        ok(".env is already fully configured")

    return True


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def _banner():
    print(_c(C.BOLD + C.CYAN, """
  ██████╗ ███████╗███████╗ █████╗ ██╗   ██╗██╗  ████████╗
  ██╔══██╗██╔════╝██╔════╝██╔══██╗██║   ██║██║  ╚══██╔══╝
  ██║  ██║█████╗  █████╗  ███████║██║   ██║██║     ██║
  ██║  ██║██╔══╝  ██╔══╝  ██╔══██║██║   ██║██║     ██║
  ██████╔╝███████╗██║     ██║  ██║╚██████╔╝███████╗██║
  ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝

  ███╗   ███╗ ██████╗ ██████╗ ███████╗
  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝
  ██╔████╔██║██║   ██║██║  ██║█████╗
  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝
  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗
  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
"""))
    print(_c(C.DIM, f"  defaultMODE Agent Setup  |  {platform.system()} {platform.machine()}"))
    print(_c(C.DIM, f"  Python {sys.version.split()[0]}  |  {ROOT}\n"))


def main():
    parser = argparse.ArgumentParser(
        description="defaultMODE setup utility",
        add_help=True,
    )
    parser.add_argument(
        "--env", action="store_true",
        help="Only configure .env keys (skip venv/install)"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Only create venv and install packages (skip .env)"
    )
    args = parser.parse_args()

    _banner()

    do_venv = not args.env
    do_env  = not args.install

    success = True

    if do_venv:
        if not _python_ok():
            sys.exit(1)
        success = setup_venv() and success
        success = install_requirements() and success

    if do_env:
        success = configure_env() and success

    hdr("Done")
    if success:
        ok("Setup complete!")
        if do_venv:
            print()
            info(f"Activate your venv with:")
            print(f"     {_c(C.YELLOW, _activate_hint())}")
        print()
    else:
        warn("Setup finished with some errors — check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
