"""Create a new experimental release from read-only provenance, with guarded patches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace(path, old, new):
    text = path.read_text(encoding='utf-8-sig')
    if text.count(old) != 1:
        raise ValueError(f'Patch anchor not unique: {path}: {old[:70]!r}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def prepare(source, destination, package):
    source, destination, package = map(lambda x: Path(x).resolve(), (source, destination, package))
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError('Destination must be a new separate directory')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    parent = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination/'CH3_PARENT_SOURCE.json').write_text(json.dumps(parent, indent=2)+'\n')
    for name in ('runtime.py','analysis.py','controller.py','remote_tests.py'):
        shutil.copy2(package/name, destination/('ch3_'+name))
    runner = destination/'run_canonical_experiment.py'
    replace(runner, 'from environment.env import Env', 'from ch3_runtime import ExperimentEnv as Env, configure as ch3_configure, compatible as ch3_compatible')
    replace(runner, '    refresh_derived_config_fields()\n', '    refresh_derived_config_fields()\n    ch3_configure(args)\n')
    replace(runner, 'BATCH_METHODS = {', 'BATCH_METHODS = {\n    "vanilla_lower_mlp_upper_uncovered_capmatched": ("vanilla_mappo", "no_attention_offload_mappo"),')
    replace(runner, 'if (contract["method"] != args.method or contract["config_hash"] != digest(snapshot)\n                or contract["source_hash"] != source["source_hash"]):', 'if not ch3_compatible(contract, snapshot, source["source_hash"], args):')
    replace(runner, '"source_hash": source["source_hash"], "files": files})', '"source_hash": source["source_hash"], "files": files, "config": snapshot})')
    replace(runner, '    return parser.parse_args(argv)', '''    parser.add_argument("--ch3-demand-radius", type=float, default=460.)
    parser.add_argument("--ch3-delay", type=int, default=0)
    parser.add_argument("--ch3-drop", type=float, default=0.)
    parser.add_argument("--ch3-uniform", action="store_true")
    parser.add_argument("--ch3-boundary", action="store_true")
    parser.add_argument("--ch3-record-requests", action="store_true")
    parser.add_argument("--ch3-actor-width", type=int, default=128)
    parser.add_argument("--ch3-critic-width", type=int, default=128)
    return parser.parse_args(argv)''')
    agents = destination/'marl_models/vanilla_mappo/agents.py'
    text = agents.read_text(); index = text.index('class VanillaCriticNetwork')
    text = text[:index].replace('config.MLP_HIDDEN_DIM','config.UPPER_MLP_ACTOR_WIDTH') + text[index:].replace('config.MLP_HIDDEN_DIM','config.UPPER_MLP_CRITIC_WIDTH')
    agents.write_text(text)
    config = destination/'config.py'
    with config.open('a') as f:
        f.write('\n# Upper-only widths; lower architecture stays at the historical shared width.\nUPPER_MLP_ACTOR_WIDTH = 128\nUPPER_MLP_CRITIC_WIDTH = 128\nCH3_DEMAND_RADIUS = 460.0\nCH3_DELAY = 0\nCH3_DROP = 0.0\n')
    print(destination)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--package',default=str(Path(__file__).parent));a=p.parse_args();prepare(a.source,a.destination,a.package)
