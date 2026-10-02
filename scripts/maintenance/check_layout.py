#!/usr/bin/env python3
"""Read-only root layout check; standard library only, no project imports."""
import argparse
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).with_name('root_allowlist.py')


def check_layout(root, allowlist):
    """Return unexpected entries; missing optional entries are allowed."""
    errors = []
    for entry in sorted(root.iterdir()):
        kind = 'directories' if entry.is_dir() else 'files'
        # .git may be a file in an isolated Git worktree.
        if entry.name == '.git' and (entry.is_dir() or entry.is_file()):
            if '.git' in allowlist['directories'] or '.git' in allowlist['files']:
                continue
        if entry.is_symlink():
            errors.append(f'{entry.name}: root symlink is not allowed')
        elif entry.name not in allowlist[kind]:
            errors.append(f'{entry.name}: unexpected root {kind}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT,
                        help='Repository root to inspect (default: this repository)')
    args = parser.parse_args()
    try:
        allowlist = runpy.run_path(str(ALLOWLIST))['ROOT_ALLOWLIST']
        for key in ('files', 'directories'):
            values = allowlist[key]
            if not isinstance(values, list) or any(
                not isinstance(v, str) or v in ('', '.', '..') or '/' in v or '\\' in v
                for v in values
            ):
                raise ValueError(f'Invalid allowlist: {key}')
        errors = check_layout(args.root, allowlist)
    except (OSError, ValueError, KeyError) as exc:
        print(f'Layout check failed: {exc}')
        return 2
    if errors:
        print('\n'.join(errors))
        print('Place new files according to AGENTS.md; do not auto-expand the allowlist.')
        return 1
    print('Root layout OK (subdirectory classification and document links require separate review).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
