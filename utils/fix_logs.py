#!/usr/bin/env python3
"""Check and repair JSONL log files with invalid UTF-8 bytes."""

import glob, os, shutil, sys, json, argparse


def find_logs(root="cache"):
    return sorted(glob.glob(f"{root}/*/logs/*.jsonl"))


def check_file(path):
    """Return (bad_byte, position) or None if file is clean."""
    try:
        open(path, encoding="utf-8").read()
        return None
    except UnicodeDecodeError as e:
        return e


def repair_file(path, dry_run=False):
    """
    Read with errors='replace', validate each line as JSON, write back clean.
    Returns (lines_total, lines_repaired, lines_dropped).
    """
    raw = open(path, encoding="utf-8", errors="replace").read()

    lines_total = 0
    lines_repaired = 0
    lines_dropped = 0
    out_lines = []

    for line in raw.splitlines():
        if not line.strip():
            continue
        lines_total += 1

        # Check if the replacement char crept in
        has_replacement = "\ufffd" in line

        try:
            obj = json.loads(line)
            if has_replacement:
                # Round-trip to drop the replacement chars from string values
                clean = json.dumps(obj, ensure_ascii=False)
                out_lines.append(clean)
                lines_repaired += 1
            else:
                out_lines.append(line)
        except json.JSONDecodeError:
            # Line is structurally broken — drop it
            lines_dropped += 1

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")

    return lines_total, lines_repaired, lines_dropped


def main():
    parser = argparse.ArgumentParser(description="Check and repair JSONL log files.")
    parser.add_argument("paths", nargs="*", help="Specific .jsonl files (default: cache/*/logs/*.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="Report issues without modifying files")
    args = parser.parse_args()

    paths = args.paths or find_logs()
    if not paths:
        print("No .jsonl files found.")
        return

    corrupt = []
    clean = []

    print(f"Checking {len(paths)} file(s)...\n")
    for path in paths:
        err = check_file(path)
        size_mb = os.path.getsize(path) / 1_048_576
        if err:
            corrupt.append((path, err))
            print(f"  CORRUPT  {path}  ({size_mb:.1f} MB)")
            print(f"           {err}")
        else:
            clean.append(path)
            print(f"  ok       {path}  ({size_mb:.1f} MB)")

    print(f"\n{len(clean)} clean, {len(corrupt)} corrupt.")

    if not corrupt:
        return

    if args.dry_run:
        print("\n--dry-run: no files modified.")
        return

    print()
    for path, _ in corrupt:
        backup = path.replace(".jsonl", "_backup.jsonl")

        # Backup
        shutil.copy2(path, backup)
        print(f"Backed up  {path}")
        print(f"        -> {backup}")

        # Repair
        total, repaired, dropped = repair_file(path)
        print(f"Repaired   {total} lines total, {repaired} cleaned, {dropped} dropped")

        # Verify
        err_after = check_file(path)
        if err_after:
            print(f"  WARNING: still corrupt after repair: {err_after}")
        else:
            print(f"  Verified clean.\n")


if __name__ == "__main__":
    main()
