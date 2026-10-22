"""Portable orchestration of the paper's frozen per-study numerical primitives.
No scheduler, site paths, historical release-controller or result selection code.
"""
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from .data import BlockEpochIterator, DataConfig, load_bin
from .models import build_model
from .optim import build_optimizer, configure_precision, cosine_lr


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def stateless_starts(seed, update, length, context, count):
    """Exact E068 one-based-update sampler, without replacement-rule changes."""
    if update < 1 or length <= context:
        raise ValueError('Sampler domain')
    return np.random.default_rng(np.random.SeedSequence([seed, update, 1068])).integers(
        0, length-context, size=count)


def batch(tokens, starts, context):
    x = np.stack([tokens[int(s):int(s)+context] for s in starts]).astype(np.int64)
    y = np.stack([tokens[int(s)+1:int(s)+context+1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def check_coefficients(model):
    net = getattr(model, 'net', model)
    if hasattr(model, 'check_coefficients'):
        model.check_coefficients()
    for block in net.blocks:
        for name in ('beta1', 'beta2'):
            if hasattr(block, name):
                b = float(getattr(block, name)().detach())
                if not 0 < b < 1:
                    raise FloatingPointError('Strict damping domain: 0 < beta < 1')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite model parameter')


@torch.no_grad()
def diagnostics(model, c, update):
    net = getattr(model, 'net', model)
    rows = []
    for j in range(c['layers']):
        block = net.blocks[j % c.get('core', c['layers'])]
        r = dict(update=update, layer=j, internal_time=j+1., forward_step=1.,
                 method=c['method'], damping_applicable=c['method'] != 'B',
                 nesterov_reference=3./(j+1))
        if c['method'] != 'B':
            for half, n in [('attn', 1), ('mlp', 2)]:
                beta = float(getattr(block, 'beta'+str(n))().detach())
                if not 0 < beta < 1:
                    raise FloatingPointError('Strict beta gate')
                r['beta_'+half] = beta
                r['gain_'+half] = float((getattr(block, 'gamma'+str(n))()+block.force_gain_floor).detach())
                r['nominal_rate_'+half] = -math.log(beta)
                r['mu_'+half] = float(getattr(block, 'mu'+str(n))().detach()) if hasattr(block, 'mu'+str(n)) else 0.
        rows.append(r)
    return rows


def data_identity(data, c, paper=True):
    locks = json.loads((Path(__file__).resolve().parents[1]/'configs/data_fingerprints.json').read_text())
    out = {}
    for split in ('train', 'val'):
        path = Path(data)/(c['dataset']+'_'+split+'.bin')
        before = path.stat()
        out[split] = dict(bytes=before.st_size, sha256=sha(path))
        if before.st_size % 2 or path.stat().st_mtime_ns != before.st_mtime_ns:
            raise ValueError('Changed/malformed uint16 corpus')
        if paper:
            if out[split] != locks[path.name]:
                raise ValueError('Corpus differs from submitted experiment: '+path.name)
            manifest = json.loads(Path(str(path)+'.manifest.json').read_text())
            if manifest['sha256'] != out[split]['sha256']:
                raise ValueError('Corpus manifest mismatch')
    return out


def evaluation_updates(c):
    if c['evaluation'] == 'legacy_post_update':
        return [i+1 for i in range(c['eval_interval'], c['updates'], c['eval_interval'])]+[c['updates']]
    return sorted({0, c['updates'], *range(c['eval_interval'], c['updates']+1, c['eval_interval'])})


@torch.no_grad()
def evaluate(model, val, it, c, device):
    state = copy.deepcopy(it.state_dict()) if it else None
    rng = rng_state()
    was_training = model.training
    model.eval()
    losses = []
    try:
        if it is None:
            starts = np.random.default_rng(c['validation_seed']).integers(
                0, len(val)-c['context'], size=c['validation_targets']//c['context'])
            batches = (batch(val, s, c['context']) for s in np.split(starts, c['eval_batches']))
        else:
            batches = (next(it) for _ in range(c['eval_batches']))
        for x, y in batches:
            loss = model(x.to(device), y.to(device))[1]
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            losses.append(float(loss))
        # Original block-epoch runners use equal-sized batch loss averages;
        # E068 accumulates target-weighted loss before division.
        if c['sampler'] == 'stateless_1068':
            weight = c['batch']*c['context']
            return sum(v*weight for v in losses)/c['validation_targets']
        return float(np.mean(losses))
    finally:
        if it:
            it.load_state_dict(state)
        restore_rng(rng)
        model.train(was_training)


def train(c, data, out, device='cuda', resume=None, stop_after=None, paper=True):
    """stop_after is interruption for tests, NEVER a shortened LR horizon."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    seed_all(c['seed'])
    configure_precision('float32', device)
    torch.use_deterministic_algorithms(c['sampler'] == 'stateless_1068')
    identity = data_identity(data, c, paper)
    model = build_model(c).to(device)
    opt = build_optimizer(model, c['peak_lr'], betas=tuple(c['betas']), scalar_lr_mult=c['scalar_lr_mult'], optimizer_name='adamw')
    train_tokens = load_bin(str(Path(data)/(c['dataset']+'_train.bin')))
    val_tokens = load_bin(str(Path(data)/(c['dataset']+'_val.bin')))
    iterator = lambda tokens, seed, split: BlockEpochIterator(tokens, DataConfig(
        block_size=c['context'], batch_size=c['batch'], grad_accum_steps=c['accumulation'], seed=seed), split)
    ti = iterator(train_tokens, c['seed'], 'train') if c['sampler'] == 'block_epoch' else None
    vi = iterator(val_tokens, c['validation_seed'], 'val') if c['sampler'] == 'block_epoch' else None
    end = c['updates'] if stop_after is None else stop_after
    if not 0 < end <= c['updates']:
        raise ValueError('Interruption outside fixed horizon')
    start = 0
    metrics, damping = [], []
    best = float('inf')
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)  # trusted own checkpoint only
        if ck['config'] != c or ck['data_identity'] != identity or ck['device_type'] != torch.device(device).type:
            raise ValueError('Resume config/data/device mismatch')
        model.load_state_dict(ck['model'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        start = ck['update']
        if not 0 < start < end:
            raise ValueError('Resume endpoint outside horizon')
        if ti:
            ti.load_state_dict(ck['train_iterator'])
            vi.load_state_dict(ck['val_iterator'])
        restore_rng(ck['rng'])
        metrics, damping = ck['metrics'], ck['damping']
        best = ck['best_val']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    write_json(out/'run_manifest.json', dict(config=c, data_identity=identity, paper_protocol=paper,
        resumed_from=str(resume) if resume else None, precision='IEEE FP32; TF32 off', python=platform.python_version(),
        torch=torch.__version__, device=device, gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        parameters=sum(p.numel() for p in model.parameters()), source_hashes={str(p.relative_to(Path(__file__).parent)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    began = time.monotonic()
    events = set(evaluation_updates(c))
    def event(update):
        metrics.append(dict(update=update, tokens=update*c['batch']*c['context']*c['accumulation'], val_nll=evaluate(model, val_tokens, vi, c, device)))
        damping.extend(diagnostics(model, c, update))
        write_csv(out/'metrics.csv', metrics)
        write_csv(out/'damping.csv', damping)
    def checkpoint(update, name, readback=False):
        state = dict(config=c, data_identity=identity, device_type=torch.device(device).type,
            update=update, best_val=best, model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
            train_iterator=ti.state_dict() if ti else None, val_iterator=vi.state_dict() if vi else None,
            metrics=metrics, damping=damping)
        tmp = out/(name+'.tmp')
        torch.save(state, tmp)
        os.replace(tmp, out/name)
        if readback:
            actual = torch.load(out/name, map_location='cpu', weights_only=False)
            if actual['update'] != update or actual['config'] != c:
                raise ValueError('Checkpoint physical readback mismatch')
            del actual
            sha(out/name)
    if start == 0 and 0 in events:
        event(0)
        if c['sampler'] == 'stateless_1068':
            checkpoint(0, 'latest.pt', readback=True)
    model.train()
    try:
        for done in range(start+1, end+1):
            schedule_step = done-1+c['learning_rate_origin']
            lr = cosine_lr(schedule_step, c['warmup_steps'], c['updates'], c['peak_lr'], c['min_lr_ratio'])
            for g in opt.param_groups:
                g['lr'] = lr*g.get('lr_mult', 1.)
            opt.zero_grad(set_to_none=True)
            if ti is None:
                indices = stateless_starts(c['seed'], done, len(train_tokens), c['context'], c['batch']*c['accumulation'])
                batches = (batch(train_tokens, s, c['context']) for s in np.split(indices, c['accumulation']))
            else:
                batches = (next(ti) for _ in range(c['accumulation']))
            for x, y in batches:
                loss = model(x.to(device), y.to(device), global_step=schedule_step)[1]/c['accumulation']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite train loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
            opt.step()
            check_coefficients(model)
            if done in events:
                event(done)
                if c['study'] in ('softmax', 'linear') and metrics[-1]['val_nll'] < best:
                    best = metrics[-1]['val_nll']
                    checkpoint(done, 'best.pt')
                if c['sampler'] == 'stateless_1068':
                    checkpoint(done, 'final.pt' if done == c['updates'] else 'latest.pt', readback=True)
            # Preserve each study's original expensive checkpoint events. Public
            # telemetry/file schema is new; timing must be remeasured, not inferred.
            if c['study'] == 'sharing' and (done % 1000 == 0 or done == c['updates']):
                checkpoint(done, 'checkpoint.pt', readback=True)
            if done == end:
                if end != c['updates'] or c['study'] in ('softmax', 'linear'):
                    checkpoint(done, 'checkpoint.pt')
                elif c['sampler'] == 'stateless_1068':
                    (out/'checkpoint.pt').symlink_to('final.pt')
        write_json(out/'summary.json', dict(complete=end==c['updates'], paper_protocol=paper, config=c,
            final_nll=metrics[-1]['val_nll'] if end==c['updates'] else None, updates=end,
            tokens=end*c['batch']*c['context']*c['accumulation'], parameters=sum(p.numel() for p in model.parameters()),
            elapsed_seconds=time.monotonic()-began, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None))
    except BaseException as e:
        write_json(out/'failure.json', dict(type=type(e).__name__, error=str(e)))
        raise
    return model, opt
