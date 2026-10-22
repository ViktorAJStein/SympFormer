"""Synthetic file fixtures only; no table outcomes or GPU measurements invented."""
import ast
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from run import presets
from sympformer.training import evaluation_updates
from analyze import collect, aggregate

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)


def fixtures(root):
    for i,c in enumerate(presets('softmax')):
        d=root/str(i)
        d.mkdir()
        times=evaluation_updates(c)
        metrics=[dict(update=u,tokens=u*16384,val_nll=7+1/(u+1)+i*.0001) for u in times]
        damping=[]
        for u in times:
            for l in range(4):
                r=dict(update=u,layer=l,internal_time=l+1,forward_step=1.,damping_applicable=c['method']!='B')
                if c['method']!='B':
                    for half in ('attn','mlp'):
                        r.update({'beta_'+half:.9,'gain_'+half:1.,'nominal_rate_'+half:-__import__('math').log(.9)})
                damping.append(r)
        (d/'summary.json').write_text(json.dumps(dict(config=c,complete=True,paper_protocol=True,updates=6103,tokens=99991552,final_nll=metrics[-1]['val_nll'],parameters=10,peak_allocated_mib=None)))
        (d/'run_manifest.json').write_text(json.dumps(dict(config=c,paper_protocol=True,data_identity={'synthetic':True},source_hashes={'synthetic':'fixture'},precision='float32',python='fixture',torch='fixture',gpu='TEST GPU')))
        write_csv(d/'metrics.csv',metrics)
        write_csv(d/'damping.csv',damping)


class ReportingTests(unittest.TestCase):
    def test_source_lineage(self):
        p=json.loads((ROOT/'docs/SOURCE_PROVENANCE.json').read_text())
        self.assertEqual(len(p['definitions']),52)
        for r in p['definitions']:
            nodes={n.name:n for n in ast.parse((ROOT/r['public_module']).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            actual=hashlib.sha256(ast.dump(nodes[r['symbol']],include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual,r['ast_sha256'])

    def test_complete_aggregation_and_adversaries(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            fixtures(root)
            rows=aggregate('softmax',root)
            self.assertEqual(len(rows),10)
            self.assertTrue(all(r['n']==5 and r['mean_allocated_gpu_hours'] is None for r in rows))
            p=root/'0/summary.json'
            original=p.read_text()
            s=json.loads(original)
            for k,v in [('complete',False),('paper_protocol',False),('tokens',100000000),('final_nll',8.)]:
                b=dict(s);b[k]=v;p.write_text(json.dumps(b))
                with self.assertRaises(ValueError):collect('softmax',root)
            p.write_text(original)
            moved=root/'0/summary-moved.json';p.rename(moved)
            with self.assertRaises(ValueError):collect('softmax',root)
            moved.rename(p)
            (root/'0/failure.json').write_text('{}')
            with self.assertRaises(ValueError):collect('softmax',root)
            (root/'0/failure.json').unlink()
            alloc=root/'0/allocation.json';alloc.write_text(json.dumps(dict(gpu_count=2,allocated_seconds=1,gpu='TEST GPU')))
            with self.assertRaises(ValueError):aggregate('softmax',root)
            alloc.unlink()
            for d in root.iterdir():
                if d.is_dir():(d/'allocation.json').write_text(json.dumps(dict(gpu_count=1,allocated_seconds=3600,gpu='TEST GPU')))
            self.assertTrue(all(r['mean_allocated_gpu_hours']==1. for r in aggregate('softmax',root)))

    def test_plot_exports(self):
        import plot
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'runs';root.mkdir();fixtures(root)
            checked=collect('softmax',root)
            # Pure plotting fixture: collection/all-seed enforcement tested above.
            block={k:v for k,v in checked.items() if k[1] in ('B','H0') and k[2] in (70020,70021,70022,71020,71021,71022)}
            with patch.object(plot,'collect',lambda *_:block):plot.plot_runs('softmax',root,Path(td)/'plots')
            f=Path(td)/'plots'
            self.assertTrue((f/'tinystories_nll.pdf').is_file())
            self.assertTrue((f/'tinystories_nll.png').is_file())
            self.assertTrue((f/'damping_raw.csv').is_file())
            self.assertTrue((f/'damping_fits.csv').is_file())
            with (f/'nll.csv').open() as h:
                rows=list(csv.DictReader(h))
            self.assertEqual(len(rows),4*62)
            self.assertEqual(int(rows[0]['update']),101)


if __name__=='__main__':
    unittest.main()
