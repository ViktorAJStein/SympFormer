"""List/dry-run/execute only the submitted paper's 340 scientific cells."""
import argparse
import copy
import json
from pathlib import Path
from sympformer.training import train

ROOT = Path(__file__).resolve().parent
STUDIES = ('softmax', 'sharing', 'linear', 'sigmoid', 'native_l2')


def presets(study):
    if study not in STUDIES:
        raise ValueError('Unreported study')
    return json.loads((ROOT/'configs'/(study+'.json')).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=STUDIES)
    p.add_argument('--index', type=int)
    p.add_argument('--list', action='store_true')
    p.add_argument('--execute', action='store_true', help='Otherwise only print the immutable protocol')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--resume', type=Path, help='Trusted own checkpoint; same full-horizon protocol')
    a = p.parse_args()
    if a.list:
        for study in ([a.study] if a.study else STUDIES):
            for i, c in enumerate(presets(study)):
                print(json.dumps(dict(study=study, index=i, method=c['method'], dataset=c['dataset'], seed=c['seed'], layout=c.get('layout'))))
        return
    if a.study is None or a.index is None or not 0 <= a.index < len(presets(a.study)):
        p.error('Provide a study and a valid index; see --list')
    c = presets(a.study)[a.index]
    print(json.dumps(c, indent=2))
    if not a.execute:
        return
    if a.data_dir is None or a.out_dir is None:
        p.error('--execute requires --data-dir and a fresh --out-dir')
    train(c, a.data_dir, a.out_dir, a.device, a.resume)


if __name__ == '__main__':
    main()
