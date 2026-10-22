"""Small CPU correctness/regression tests, never a quality experiment."""
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sympformer.core import CausalSelfAttention, ModelConfig
from sympformer.models import build_model
from sympformer.mobility import MobilityAttention, MobilityConfig, tensor_metric
from sympformer.softmax import attention_connection
from sympformer.linear import LinearHBConfig, project, linear_proxy
from sympformer.training import (train, seed_all, stateless_starts, batch, evaluation_updates,
                                 rng_state, check_coefficients)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def cell(study='softmax', method='R2'):
    c = next(c for c in json.loads((ROOT/'configs'/(study+'.json')).read_text()) if c['method'] == method)
    c = copy.deepcopy(c)
    c.update(width=8, heads=2, layers=8 if study == 'sharing' else 2, context=8, vocab=41,
             batch=2, accumulation=2, updates=4, eval_interval=2, eval_batches=2,
             validation_targets=32, warmup_steps=1, tokens=128)
    return c


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


class PaperTests(unittest.TestCase):
    def test_matrix(self):
        expected = dict(softmax=50, sharing=200, linear=30, sigmoid=30, native_l2=30)
        for study, count in expected.items():
            cs = json.loads((ROOT/'configs'/(study+'.json')).read_text())
            self.assertEqual(len(cs), count)
            self.assertEqual(len({(c['method'], c['dataset'], c['seed'], c.get('core')) for c in cs}), count)
            for c in cs:
                self.assertEqual(c['updates']*c['batch']*c['accumulation']*c['context'], 99991552)
                self.assertEqual(c['eval_batches']*c['batch']*c['context'], 32768)
                self.assertTrue(c['final_endpoint_only'])
                self.assertEqual(c['precision'], 'float32')
                events = evaluation_updates(c)
                self.assertEqual(events[-1], 6103)
                self.assertEqual(events[0], 101 if study == 'softmax' else 0)
                self.assertEqual(len(events), 62 if study == 'softmax' else 63)

    def test_native_parameter_counts(self):
        # Counts at the actual paper shape; no training/forward pass.
        expected = {'softmax': {'B': 3424832, 'Y': 6652760, 'H0': 6652756, 'H1': 6652756, 'R2': 6652756},
                    'linear': {'B': 3449408, 'Y': 6701912, 'H0': 6701908, 'H1': 6701908, 'R2': 6701908},
                    'sigmoid': {'B': 3449408, 'Y': 6701912, 'H0': 6701908, 'H1': 6701908, 'R2': 6701908},
                    'native_l2': {'B': 3449408, 'Y': 6701912, 'H0': 6701908, 'H1': 6701908, 'R2': 6701908}}
        for study, counts in expected.items():
            cs = json.loads((ROOT/'configs'/(study+'.json')).read_text())
            for method, expected_count in counts.items():
                seed_all(1)
                # Exact full-shape count without repeatedly allocating large
                # embedding tensors. Real tiny forward/gradient tests follow.
                with torch.device('meta'):
                    model = build_model(next(c for c in cs if c['method'] == method))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
                del model

    def test_sharing_counts_and_initial_equality(self):
        cs = json.loads((ROOT/'configs/sharing.json').read_text())
        expected = {'B': (14583040, 19304704), 'Y': (27592460, 32315696),
                    'H0': (27592458, 32315688), 'H1': (27592458, 32315688),
                    'R2': (27592458, 32315688)}
        for method, counts in expected.items():
            for core, count in zip((2, 8), counts):
                with torch.device('meta'):
                    model = build_model(next(c for c in cs if c['method']==method and c['core']==core))
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                del model
            c = cell('sharing', method)
            a = build_model(c).double()
            c['core'] = 8
            c['layout'] = 'untied'
            b = build_model(c).double()
            x = torch.arange(8).reshape(1, 8)
            self.assertTrue(torch.equal(a(x)[0], b(x)[0]), method)

    def test_causality_gradients(self):
        for study in ('softmax', 'sharing', 'linear', 'sigmoid', 'native_l2'):
            for method in ('B', 'Y', 'H0', 'H1', 'R2'):
                c = cell(study, method)
                seed_all(3)
                model = build_model(c).double()
                x = torch.arange(8).reshape(1, 8) % 41
                y = x.roll(-1, 1)
                logits, loss = model(x, y)
                changed = x.clone()
                changed[:, 4:] = (changed[:, 4:]+11) % 41
                other = model(changed)[0]
                self.assertTrue(torch.allclose(logits[:, :4], other[:, :4], atol=1e-10, rtol=0), (study, method))
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                check_coefficients(model)

    def test_softmax_and_sigmoid_formulas(self):
        seed_all(91)
        attn = CausalSelfAttention(ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=3, vocab_size=11)).double()
        z, u = torch.randn(1, 3, 4, dtype=torch.double), torch.randn(1, 3, 4, dtype=torch.double)
        for family in ('softmax', 'sigmoid_normalized'):
            module = attn if family == 'softmax' else MobilityAttention(copy.deepcopy(attn), MobilityConfig(family, 'H1'))
            actual_f, actual_g = attention_connection(module, z, u) if family == 'softmax' else module.force_and_connection(z, u)[:2]
            q, k, w = [a.view(1, 3, 2, 2).transpose(1, 2) for a in module.c_attn(z).chunk(3, -1)]
            dq, dk = [a.view(1, 3, 2, 2).transpose(1, 2) for a in torch.nn.functional.linear(u, module.c_attn.weight[:8]).chunk(2, -1)]
            force, corr = [], []
            for i in range(3):
                hs, ds, drift = [], [], []
                for h in range(2):
                    l = q[0, h, i] @ k[0, h, :i+1].T / math.sqrt(2)
                    p = l.softmax(-1) if family == 'softmax' else l.sigmoid()/l.sigmoid().sum()
                    e = p if family == 'softmax' else p*torch.sigmoid(-l)
                    direction = (dq[0, h, i] @ k[0, h, :i+1].T + q[0, h, i] @ dk[0, h, :i+1].T)/math.sqrt(2)
                    drift.append((e*direction).sum())
                    hs.append(p @ w[0, h, :i+1])
                    speed = u[0, :i+1].square().sum(-1)
                    ds.append(u[0, i].square().sum()*(e @ w[0, h, :i+1]) + (e*speed) @ w[0, h, :i+1])
                force.append(module.c_proj(torch.cat(hs)))
                corr.append(torch.stack(drift).mean()*u[0, i]-.5*module.c_proj(torch.cat(ds)))
            self.assertTrue(torch.allclose(actual_f, torch.stack(force)[None], atol=1e-12, rtol=1e-10))
            self.assertTrue(torch.allclose(actual_g, torch.stack(corr)[None], atol=1e-12, rtol=1e-10))

    def test_linear_ridge_formula(self):
        seed_all(9)
        attn = CausalSelfAttention(ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=3, vocab_size=11)).double()
        z, u = torch.randn(1, 3, 4, dtype=torch.double), torch.randn(1, 3, 4, dtype=torch.double)
        q, k, _ = project(attn, z)
        hb = LinearHBConfig(mode='H1_L', chunk=2, checkpoint_chunks=False)
        actual = linear_proxy(attn, q, k, u, hb)[0]
        d = torch.nn.functional.linear(u, attn.c_attn.weight[:8]);dq, dk = [v.reshape(1, 3, 2, 2).transpose(1, 2)/2**.25 for v in d.chunk(2, -1)]
        out = []
        for i in range(3):
            heads = []
            for h in range(2):
                Q, K, U = q[0, h, :i+1], k[0, h, :i+1], u[0, :i+1]
                pq = torch.linalg.inv(Q.T @ Q+.001*torch.eye(2));pk = torch.linalg.inv(K.T @ K+.001*torch.eye(2))
                E = pq @ Q.T @ U;J = pk @ K.T @ dk[0, h, :i+1];T = pk @ K.T @ Q
                wk = attn.c_attn.weight[4:8].reshape(2, 2, 4)[h]/2**.25
                heads.append(-dq[0, h, i] @ (K.T @ K @ pk) @ E - q[0, h, i] @ J.T @ E + u[0, i] @ E.T @ T @ wk)
            out.append(torch.stack(heads).mean(0))
        self.assertTrue(torch.allclose(actual, torch.stack(out)[None], atol=1e-10, rtol=1e-8))

    def test_native_l2_tensor_derivative(self):
        seed_all(17)
        attn = MobilityAttention(CausalSelfAttention(ModelConfig(n_embd=4, n_head=2, n_layer=1, block_size=3, vocab_size=11)), MobilityConfig('l2_native_tensor_proxy', 'H1')).double()
        z = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
        u = torch.randn_like(z)
        f, actual, _ = attn.force_and_connection(z, u)
        expected = []
        for i in range(3):
            zp, up = z[:, :i+1], u[:, :i+1]
            def g(x):
                force = attn.force_and_connection(x)[0]
                return tensor_metric(force)[:, -1]
            metric, dotg = torch.autograd.functional.jvp(g, zp, up, create_graph=True)
            ui = up[:, -1]
            energy = torch.einsum('bi,bij,bj->', ui, g(zp), ui)
            selfgrad = torch.autograd.grad(energy, zp, create_graph=True)[0][:, -1]
            covector = (dotg @ ui[..., None]).squeeze(-1)-.5*selfgrad
            fi = f[:, i];a = torch.nn.functional.softplus(fi[:, 0])+.001
            expected.append(covector+fi*(fi*covector).sum(-1, keepdim=True)/a[:, None])
        self.assertTrue(torch.allclose(actual, torch.stack(expected, 1), atol=1e-10, rtol=1e-8))
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_training_and_exact_resume(self):
        for study in ('softmax', 'sharing', 'linear', 'sigmoid', 'native_l2'):
            with tempfile.TemporaryDirectory() as td:
                td = Path(td);c = cell(study)
                for split in ('train', 'val'):
                    (np.arange(4096, dtype=np.uint16) % 41).tofile(td/(c['dataset']+'_'+split+'.bin'))
                full, opt = train(c, td, td/'full', 'cpu', paper=False)
                train(c, td, td/'half', 'cpu', stop_after=2, paper=False)
                resumed, opt2 = train(c, td, td/'resumed', 'cpu', resume=td/'half/checkpoint.pt', paper=False)
                self.assertTrue(equal(full.state_dict(), resumed.state_dict()), study)
                self.assertTrue(equal(opt.state_dict(), opt2.state_dict()), study)
                a = torch.load(td/'full/checkpoint.pt', weights_only=False)
                b = torch.load(td/'resumed/checkpoint.pt', weights_only=False)
                for key in ('rng', 'metrics', 'damping', 'train_iterator', 'val_iterator'):
                    self.assertTrue(equal(a[key], b[key]), (study, key))
                bad = copy.deepcopy(c);bad['seed'] += 1
                with self.assertRaises(ValueError):
                    train(bad, td, td/'bad', 'cpu', resume=td/'half/checkpoint.pt', paper=False)

    def test_negative_domains(self):
        with self.assertRaises(ValueError):
            MobilityConfig('sinkhorn_prefix', 'R2')
        with self.assertRaises(ValueError):
            LinearHBConfig(ridge=.01)
        with self.assertRaises(ValueError):
            stateless_starts(1, 0, 32, 8, 4)
        c = cell('softmax', 'B');model = build_model(c)
        from sympformer.training import data_identity
        with tempfile.TemporaryDirectory() as td:
            np.arange(64, dtype=np.uint16).tofile(Path(td)/(c['dataset']+'_train.bin'))
            with self.assertRaises(ValueError):
                data_identity(td, c, paper=True)


if __name__ == '__main__':
    unittest.main()
