"""Dispatch reviewed v4 presets to isolated frozen runtimes; no Slurm submission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def verify_sources():
    p=json.loads((ROOT/'protocols.json').read_text())
    for n,h in {**p['core_sha256'],**p['config_sha256']}.items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h, 'Changed snapshot: '+n
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--config',help='Preset path relative to paper_v4, e.g. configs/softmax_primary/R2_tinystories_s70020.json')
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--execute',action='store_true',help='Without this flag, print the command only')
    ap.add_argument('--resume',type=Path,help='Trusted checkpoint, same preset/horizon only')
    args=ap.parse_args();p=verify_sources()
    if args.list:
        for cell in p['cells']:print(cell['study'],cell['method'],cell['dataset'],cell['seed'],cell['config'])
        return
    if not args.config:ap.error('--config or --list is required')
    cell=next((c for c in p['cells'] if c['config']==args.config),None)
    if cell is None:ap.error('Unknown preset; select a listed config')
    if args.data_dir is None or args.out_dir is None:ap.error('--data-dir and --out-dir are required')
    runtime=ROOT/cell['runtime']
    if cell['runtime']=='looped':
        cmd=[sys.executable,str(runtime/'scripts/train_looped_hb.py'),'--config',str(ROOT/args.config),
             '--data',str(args.data_dir.resolve()),'--out',str(args.out_dir.resolve()),'--device',args.device]
    else:
        cmd=[sys.executable,str(runtime/'train.py'),'--config',str(ROOT/args.config),'--data_dir',str(args.data_dir.resolve()),
             '--out_dir',str(args.out_dir.resolve()),'--device',args.device]
    if args.resume:cmd+=['--resume',str(args.resume.resolve())]
    print(json.dumps(dict(study=cell['study'],command=cmd,actual_tokens=p['actual_tokens'],execute=args.execute),indent=2),flush=True)
    if args.execute:subprocess.run(cmd,cwd=runtime,check=True)


if __name__=='__main__':main()
