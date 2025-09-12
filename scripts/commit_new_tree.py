#!/usr/bin/env python3
"""
Create a commit whose tree is the provided tree SHA and update a branch ref to that new commit.
Usage: python scripts/commit_new_tree.py <root_tree_sha> <branch>
"""
import sys
import subprocess


def main(argv):
    if len(argv) < 3:
        print("Usage: commit_new_tree.py <root_tree_sha> <branch>")
        return 2
    root = argv[1]
    branch = argv[2]
    msg = "Remove reserved file 'nul' from tree (automated)\n\nAutomated commit created to remove reserved filename 'nul'."
    try:
        p = subprocess.run(["git", "commit-tree", root, "-p", "HEAD"], input=msg.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('commit-tree rc=', p.returncode)
        print('stdout:\n', p.stdout.decode())
        print('stderr:\n', p.stderr.decode())
        if p.returncode != 0:
            return 3
        new = p.stdout.decode().strip()
        print('new commit:', new)
        p2 = subprocess.run(["git", "update-ref", f"refs/heads/{branch}", new], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('update-ref rc=', p2.returncode)
        print('update-ref stdout\n', p2.stdout.decode())
        print('update-ref stderr\n', p2.stderr.decode())
        p3 = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
        print('HEAD now:', p3.stdout.decode().strip())
        return 0
    except Exception as e:
        print('EXCEPTION', e)
        return 4


if __name__ == '__main__':
    sys.exit(main(sys.argv))

