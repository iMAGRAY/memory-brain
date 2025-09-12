#!/usr/bin/env python3
"""
Remove reserved filename 'nul' from the tip commit by creating a new commit
whose tree excludes the path 'nul'.

This script does NOT rewrite history (it creates a new commit on the current
branch pointing to the new tree and sets the branch to that commit). It does
not modify the working tree files. Use with caution.

Designed to run in repository root. Produces verbose logs to stdout.
"""
from __future__ import annotations
import subprocess
import sys
import os


def run(cmd, input_bytes=None, check=True):
    print("+ RUN:", " ".join(cmd))
    p = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p.stdout.decode("utf-8", errors="replace")
    stderr = p.stderr.decode("utf-8", errors="replace")
    print("+ rc=", p.returncode)
    if stdout:
        print("+ stdout:\n", stdout)
    if stderr:
        print("+ stderr:\n", stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nrc={p.returncode}\nstderr={stderr}")
    return p.stdout


def main():
    # Ensure we're in a git repo
    try:
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception as e:
        print("ERROR: not a git repo or git not available:", e)
        sys.exit(1)

    if not branch or branch == "HEAD":
        print("Note: detached HEAD or unknown branch. New commit will be created but not attached to a named branch.")
        branch = None
    else:
        print("Current branch:", branch)

    parent = run(["git", "rev-parse", "HEAD"]).decode().strip()
    print("Parent commit:", parent)

    # Get ls-tree entries (NUL-separated)
    entries_bytes = subprocess.check_output(["git", "ls-tree", "-r", "-z", "HEAD"])  # may raise
    parts = entries_bytes.split(b"\x00")

    # Parse entries and exclude exact path 'nul'
    file_entries = []  # list of tuples (mode, type, sha, path_bytes)
    for p in parts:
        if not p:
            continue
        if b"\t" not in p:
            # malformed, skip
            print("WARN: skipping malformed ls-tree entry:", p)
            continue
        left, path = p.split(b"\t", 1)
        left_parts = left.split()
        if len(left_parts) != 3:
            print("WARN: unexpected left part for entry:", left)
            continue
        mode = left_parts[0].decode()
        typ = left_parts[1].decode()
        sha = left_parts[2].decode()
        # skip exact top-level path 'nul'
        if path == b"nul" or path.decode("utf-8", errors="replace") == "nul":
            print("Skipping reserved path (excluded from new tree):", path)
            continue
        file_entries.append((mode, typ, sha, path))

    if not file_entries:
        print("No files found (after exclusion). Aborting.")
        sys.exit(1)

    # Build directory -> entries mapping, entries are (mode,type,sha,name)
    entries_for_dir = {}
    dirs = set()
    for mode, typ, sha, path in file_entries:
        # convert path bytes -> normalized unix-style string
        path_str = path.decode("utf-8", errors="surrogateescape")
        dirname, basename = os.path.split(path_str)
        # represent root as empty string
        entries_for_dir.setdefault(dirname, []).append((mode, typ, sha, basename))
        dirs.add(dirname)

    # ensure all parent directories exist in the map
    to_add = set(dirs)
    for d in list(dirs):
        parent = d
        while parent:
            parent = os.path.dirname(parent)
            if parent not in to_add:
                to_add.add(parent)
                entries_for_dir.setdefault(parent, [])

    dirs = list(to_add)
    # sort directories by descending depth (deeper first)
    dirs.sort(key=lambda x: x.count("/"), reverse=True)

    tree_shas = {}

    for d in dirs:
        entries = entries_for_dir.get(d, [])
        # stable sort by name
        entries_sorted = sorted(entries, key=lambda it: it[3])
        mktree_input_lines = []
        for mode, typ, sha, name in entries_sorted:
            # form: <mode> <type> <sha>\t<name>\n
            # Note: name should be basename only (no slashes)
            if "\t" in name or "\n" in name:
                raise RuntimeError(f"Unsupported file name containing control chars: {name}")
            line = f"{mode} {typ} {sha}\t{name}\n"
            mktree_input_lines.append(line)

        mktree_input = "".join(mktree_input_lines).encode("utf-8")
        print(f"Creating tree for directory: '{d}' with {len(entries_sorted)} entries")
        try:
            tree_out = run(["git", "mktree"], input_bytes=mktree_input)
        except Exception as e:
            print("ERROR: git mktree failed for directory:", d)
            raise
        tree_sha = tree_out.decode().strip()
        print("-> tree sha:", tree_sha)
        tree_shas[d] = tree_sha
        # add this tree as an entry to its parent
        if d != "":
            parent = os.path.dirname(d)
            base = os.path.basename(d)
            entries_for_dir.setdefault(parent, []).append(("040000", "tree", tree_sha, base))

    root_sha = tree_shas.get("")
    if not root_sha:
        print("ERROR: failed to create root tree")
        sys.exit(1)

    print("Root tree sha:", root_sha)

    # Create new commit with parent HEAD
    commit_message = "Remove reserved file 'nul' from tree (automated)"
    print("Creating commit with parent HEAD...")
    p = subprocess.run(["git", "commit-tree", root_sha, "-p", parent], input=commit_message.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print("ERROR: git commit-tree failed")
        print(p.stderr.decode("utf-8", errors="replace"))
        sys.exit(1)
    new_commit = p.stdout.decode().strip()
    print("New commit created:", new_commit)

    # Update branch ref if named
    if branch:
        print(f"Updating branch '{branch}' to point to new commit {new_commit}")
        run(["git", "update-ref", f"refs/heads/{branch}", new_commit])
        print("Branch updated.")
    else:
        print("Detached HEAD: new commit created but branch not updated.")

    print("Operation complete. New HEAD commit:", new_commit)
    print("Now running 'git status --porcelain' to show working tree status:")
    run(["git", "status", "--porcelain"])  # show status


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL:", e)
        sys.exit(2)

