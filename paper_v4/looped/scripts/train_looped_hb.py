"""Isolated E063 fixed-budget trainer; no launch, no historical source mutation."""
import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import DataConfig, BlockEpochIterator, load_bin
from model import ModelConfig
from train import build_optimizer, configure_precision, cosine_lr
from scripts.looped_hb_model import LoopedDecoder, METHODS, LAYOUTS, VERSION
from scripts.looped_hb_telemetry import Observer


@dataclass(frozen=True)
class Protocol:
    method: str = 'B'
    layout: str = 'tied'
    dataset: str = 'tinystories'
    seed: int = 97020
    validation_seed: int = 97018
    width: int = 256
    heads: int = 8
    context: int = 512
    vocab: int = 50304
    batch: int = 4
    accumulation: int = 8
    updates: int = 6103
    eval_interval: int = 100
    eval_batches: int = 16
    peak_lr: float = .0006
    scalar_lr_mult: float = 10.
    warmup_ratio: float = .1
    min_lr_ratio: float = .1
    grad_clip: float = 1.
    version: str = VERSION
    quality_approved: bool = False

    def __post_init__(self):
        if self.method not in METHODS or self.layout not in LAYOUTS or self.version != VERSION:
            raise ValueError('Invalid E063 method/layout/version')
        if self.dataset not in ('tinystories', 'openwebtext'):
            raise ValueError('Invalid dataset')
        for key in ('width', 'heads', 'context', 'vocab', 'batch', 'accumulation', 'updates', 'eval_interval', 'eval_batches'):
            if type(getattr(self,key)) is not int or getattr(self,key) <= 0:
                raise ValueError('Invalid positive integer: '+key)
        if self.width % self.heads:
            raise ValueError('Width/head mismatch')
        if not (0 <= self.warmup_ratio < 1 and 0 < self.min_lr_ratio <= 1):
            raise ValueError('Invalid schedule')
        if not all(math.isfinite(x) and x > 0 for x in (self.peak_lr, self.scalar_lr_mult, self.grad_clip)):
            raise ValueError('Invalid optimizer')

    @property
    def tokens_per_update(self):
        return self.batch*self.context*self.accumulation


def make_model(c):
    cfg = ModelConfig(n_layer=8, n_head=c.heads, n_embd=c.width, block_size=c.context,
                      vocab_size=c.vocab, dropout=0., bias=False,
                      yurii_force_gain_floor=0. if c.method == 'B' else 1e-6)
    return LoopedDecoder(cfg, c.method, c.layout, c.seed)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@torch.no_grad()
def evaluate(model, iterator, c, device, loops=4):
    state = copy.deepcopy(iterator.state_dict())
    rng = rng_state()
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(c.eval_batches):
            x, y = next(iterator)
            loss = model(x.to(device), y.to(device), loops=loops)[1]
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            losses.append(float(loss))
    finally:
        iterator.load_state_dict(state)
        restore_rng(rng)
        model.train(was_training)
    return float(np.mean(losses))


@torch.no_grad()
def diagnostics(model, tokens, c, device, update):
    was_training = model.training
    rng = rng_state()
    model.eval()
    try:
        ids = torch.from_numpy(np.array(tokens[:2*c.context], dtype=np.int64).reshape(2,c.context)).to(device)
        model(ids, capture=True)
        rows = [dict(update=update, **r) for r in model.last_trace]
        for row in rows:
            if not all(v is None or not isinstance(v, float) or math.isfinite(v) for v in row.values()):
                raise FloatingPointError('Nonfinite diagnostic')
        return rows
    finally:
        restore_rng(rng)
        model.train(was_training)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run(c, data_dir, out, device='cpu', stop_after=None, resume=None, inference=False):
    """stop_after preserves the full schedule; resume writes to a fresh directory."""
    if stop_after is not None and not 0 < stop_after <= c.updates:
        raise ValueError('Invalid interruption endpoint')
    end = c.updates if stop_after is None else stop_after
    # No private cluster claim; validate actual file bytes before training/resume.
    claim = None
    for split in ('train','val'):
        path = Path(data_dir)/(c.dataset+'_'+split+'.bin')
        before = path.stat(); digest = sha(path); after = path.stat()
        assert before.st_mtime_ns == after.st_mtime_ns and before.st_size == after.st_size
        assert after.st_size % 2 == 0 and after.st_size > 2*c.context
        manifest = Path(str(path)+'.manifest.json')
        if manifest.exists():
            assert json.loads(manifest.read_text())['sha256'] == digest, 'Dataset manifest/physical bytes mismatch'
        elif c.vocab == 50304:
            raise ValueError('Paper data requires a manifest sidecar')
    seed_all(c.seed); configure_precision('float32', device)
    model = make_model(c).to(device)
    opt = build_optimizer(model, c.peak_lr, betas=(.9,.95), scalar_lr_mult=c.scalar_lr_mult, optimizer_name='adamw')
    params = [p for group in opt.param_groups for p in group['params']]
    assert len(params) == len({id(p) for p in params})
    train_tokens = load_bin(str(Path(data_dir)/(c.dataset+'_train.bin')))
    val_tokens = load_bin(str(Path(data_dir)/(c.dataset+'_val.bin')))
    dc = DataConfig(block_size=c.context, batch_size=c.batch, grad_accum_steps=c.accumulation, seed=c.seed)
    train_it = BlockEpochIterator(train_tokens, dc, 'train')
    val_it = BlockEpochIterator(val_tokens, DataConfig(block_size=c.context,batch_size=c.batch,seed=c.validation_seed), 'val')
    data_identity = {split:dict(bytes=len(tokens)*2) for split,tokens in (('train',train_tokens),('val',val_tokens))}
    # Physical hashes are checked by the GPU pilot before opening these files.
    # Fixtures receive their own content identity too, preventing same-sized replacements.
    for split in data_identity:
        path=Path(data_dir)/(c.dataset+'_'+split+'.bin')
        manifest=Path(str(path)+'.manifest.json')
        data_identity[split]['sha256']=json.loads(manifest.read_text())['sha256'] if manifest.exists() else sha(path)
    start = 0; metrics = []; traces = []; training_calls = 0; evaluation_calls = 0
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        if ck['protocol'] != asdict(c) or ck['data_identity'] != data_identity:
            raise ValueError('Incompatible E063 resume protocol/data')
        if ck['device_type'] != torch.device(device).type:
            raise ValueError('Incompatible resume device cohort')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        train_it.load_state_dict(ck['train_iterator']); val_it.load_state_dict(ck['val_iterator'])
        start=ck['update'];metrics=ck['metrics'];traces=ck['traces']
        training_calls=ck['training_calls'];evaluation_calls=ck['evaluation_calls']
        restore_rng(ck['rng'])
        if not start < end:
            raise ValueError('Resume does not advance training')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    observer=Observer(out,c,model,claim,ck['observer'] if resume else None)
    validation_signature=observer.validation_identity(val_it,c)
    (out/'run_manifest.json').write_text(json.dumps(observer.manifest(asdict(c),data_identity,model),indent=2)+'\n')
    started=time.monotonic()
    def event(update):
        nonlocal evaluation_calls
        loss=evaluate(model,val_it,c,device)
        evaluation_calls += c.eval_batches
        metrics.append(dict(update=update,tokens=update*c.tokens_per_update,val_nll=loss))
        traces.extend(diagnostics(model,val_tokens,c,device,update))
        save_csv(out/'metrics.csv',metrics);save_csv(out/'dynamics.csv',traces)
    if start==0:
        event(0)
    model.train(); model.check_coefficients()
    train_times=[]
    for update in range(start,end):
        lr=cosine_lr(update,round(c.warmup_ratio*c.updates),c.updates,c.peak_lr,c.min_lr_ratio)
        for group in opt.param_groups:
            group['lr']=lr*group.get('lr_mult',1.)
        if device.startswith('cuda'):torch.cuda.synchronize()
        begin=time.monotonic();opt.zero_grad(set_to_none=True)
        for _ in range(c.accumulation):
            x,y=next(train_it)
            observer.sample(x,y)
            loss=model(x.to(device),y.to(device))[1]/c.accumulation
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();training_calls+=1
        torch.nn.utils.clip_grad_norm_(model.parameters(),c.grad_clip,error_if_nonfinite=True)
        opt.step();model.check_coefficients()
        if device.startswith('cuda'):torch.cuda.synchronize()
        train_times.append(time.monotonic()-begin)
        observer.after_update(model,update+1,train_times[-1],lr)
        train_times[-1]=time.monotonic()-begin
        observer.updates[-1]['optimizer_seconds']=train_times[-1]
        if (update+1)%c.eval_interval==0 or update+1==c.updates:
            event(update+1)
    checkpoint=dict(protocol=asdict(c),model=model.state_dict(),opt=opt.state_dict(),update=end,
        train_iterator=train_it.state_dict(),val_iterator=val_it.state_dict(),rng=rng_state(),
        metrics=metrics,traces=traces,training_calls=training_calls,evaluation_calls=evaluation_calls,
        data_identity=data_identity,device_type=torch.device(device).type,observer=observer.state_dict())
    begin=time.monotonic();torch.save(checkpoint,out/'checkpoint.pt');checkpoint_seconds=time.monotonic()-begin
    loop_rows=[]
    if inference and end==c.updates:
        for loops in ((1,2,4,8,16) if c.layout=='tied' else (4,)):
            begin=time.monotonic()
            try:
                value=evaluate(model,val_it,c,device,loops)
                status='PASS';error=None
            except FloatingPointError as exc:
                if loops==4:raise
                value=None;status='NONFINITE_EXTRA_DEPTH';error=str(exc)
            loop_rows.append(dict(loops=loops,logical_blocks=2*loops,val_nll=value,status=status,error=error,
                                  validation_seconds=time.monotonic()-begin,trained_depth=loops==4,
                                  validation_signature=validation_signature))
        save_csv(out/'inference.csv',loop_rows)
    result=dict(status='PASS',quality_run=False,method=c.method,layout=c.layout,update=end,
        complete=end==c.updates,tokens=end*c.tokens_per_update,training_calls=training_calls,
        evaluation_calls=evaluation_calls,parameters=sum(p.numel() for p in model.parameters()),
        unique_blocks=len(model.net.blocks),logical_blocks=8,train_step_seconds=train_times,
        elapsed_seconds=time.monotonic()-started,checkpoint_seconds=checkpoint_seconds,
        final_val=metrics[-1]['val_nll'] if end==c.updates else None)
    result=observer.finish(result,model,end==c.updates,loop_rows)
    result['elapsed_seconds']=time.monotonic()-started
    result['tokens_per_second']=result['tokens']/max(result['elapsed_seconds'],1e-30)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--stop-after',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--inference',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    c=Protocol(**json.loads(a.config.read_text()))
    try:
        result=run(c,a.data,a.out,a.device,a.stop_after,a.resume,a.inference)
    except Exception as error:
        if a.out.exists():
            (a.out/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)))+'\n')
        raise
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
