"""Explicit root allowlist; change only for an authorized layout change.

Includes existing entries and known local test caches; missing entries are allowed.
Python format keeps this configuration in existing source snapshots.
"""

ROOT_ALLOWLIST = {
    'version': 1,
    'files': [
        '.DS_Store',
        '.gitignore',
        '.zcodeignore',
        'AGENTS.md',
        'LICENSE',
        'README.md',
        'analyze_canonical_suite.py',
        'config.py',
        'paths.py',
        'run_canonical_experiment.py',
        'run_chapter4_suite.py',
        'run_hierarchical_mappo_experiment.py',
        'run_remote_python.sh',
        'wodexuexi.code-workspace',
    ],
    'directories': [
        '.claude',
        '.git',
        '.mypy_cache',
        '.plot-venv',
        '.pytest_cache',
        '.spec-workflow',
        '.tmp',
        '.venv',
        '.vscode',
        '.zcode',
        '__pycache__',
        'docs',
        'environment',
        'latex',
        'logs',
        'marl_models',
        'output',
        'recent',
        'results',
        'saved_models',
        'scripts',
        'tasks',
        'tests',
        'tools',
        'train_logs',
        'utils',
    ],
}
