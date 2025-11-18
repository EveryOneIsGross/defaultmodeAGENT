#!/usr/bin/env python3
import os, sys, argparse
from pathlib import Path

def load_ignore_sets(root):
    ig_files, ig_dirs = set(), set()
    p = root/'.gitignore'
    if not p.is_file(): return ig_files, ig_dirs
    lines = [l.rstrip() for l in p.read_text(encoding='utf-8', errors='ignore').splitlines()]
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith('#'): continue
        neg = s.startswith('!')
        if neg: s = s[1:].strip()
        s = s.lstrip('./')
        dir_only = s.endswith('/')
        if dir_only: s = s[:-1]
        anchored = s.startswith('/')
        if anchored: s = s[1:]
        pat = s if anchored else ('**/'+s)
        hits = list(root.glob(pat))
        if dir_only: hits = [h for h in hits if h.is_dir()]
        if neg:
            for h in hits:
                rel = h.relative_to(root).as_posix()
                (ig_dirs if h.is_dir() else ig_files).discard(rel)
        else:
            for h in hits:
                rel = h.relative_to(root).as_posix()
                (ig_dirs if h.is_dir() else ig_files).add(rel)
    # Always hide the VCS dir
    if (root/'.git').is_dir(): ig_dirs.add('.git')
    return ig_files, ig_dirs

def is_ignored(rel_posix, ig_files, ig_dirs):
    if rel_posix in ig_files or rel_posix in ig_dirs: return True
    parts = rel_posix.split('/')
    for i in range(1, len(parts)):
        if '/'.join(parts[:i]) in ig_dirs: return True
    return False

def tree(start, dirs_only=False):
    root = Path(start).resolve()
    label = '.' if root == Path('.').resolve() else root.name
    print(label)
    ig_files, ig_dirs = load_ignore_sets(root)

    def walk(d, pref=''):
        try:
            entries = sorted(os.scandir(d), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            return
        kept = []
        for e in entries:
            isdir = e.is_dir(follow_symlinks=False)
            rel = Path(e.path).resolve().relative_to(root).as_posix()
            if is_ignored(rel, ig_files, ig_dirs): 
                continue
            if dirs_only and not isdir: 
                continue
            kept.append((e, isdir))
        for i, (e, isdir) in enumerate(kept):
            last = i == len(kept)-1
            print(pref + ('└── ' if last else '├── ') + e.name + ('/' if isdir else ''))
            if isdir:
                walk(e.path, pref + ('    ' if last else '│   '))
    walk(root)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog='git_tree', description='ASCII tree respecting .gitignore (no extra deps)')
    ap.add_argument('path', nargs='?', default='.', help='start path (default: .)')
    ap.add_argument('-d','--dirs-only', action='store_true', help='show directories only')
    args = ap.parse_args()
    tree(args.path, args.dirs_only)
