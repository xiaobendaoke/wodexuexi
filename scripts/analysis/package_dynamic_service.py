"""Package a complete isolated Python release without Git or historical data."""
import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paths import REPO_ROOT
ROOT = REPO_ROOT
from scripts.analysis.dynamic_service_experiment import source_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.tmp/dynamic_service/release.zip')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in source_contract():
            archive.write(ROOT/relative, relative)
    print(args.output)


if __name__ == '__main__':
    main()
