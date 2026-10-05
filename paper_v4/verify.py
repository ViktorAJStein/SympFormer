"""CPU-only correctness verification for the isolated paper presets.
Artifacts go outside the source tree. No GPU allocation or decision-scale training.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run import ROOT,verify_sources


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);p=verify_sources()
    assert p['counts']==dict(linear_primary=30,linear_statistical_controls=12,softmax_lookahead=32,softmax_primary=50,softmax_looped=48,softmax_shapes=120,softmax_strength=12,softmax_width256=24)
    for cell in p['cells']:
        c=json.loads((ROOT/cell['config']).read_text())
        assert not any(k in c for k in ('data_dir','out_dir','resume','config','device'))
        if cell['runtime']=='looped':
            assert (c['width'],c['heads'],c['context'],c['vocab'])==(256,8,512,50304)
            assert c['updates']==6103 and c['batch']*c['accumulation']*c['context']==16384
            assert c['validation_seed']==97018 and c['scalar_lr_mult']==10 and c['peak_lr']==.0006
            continue
        assert c['n_embd'] in (64,256,512) and c['vocab_size']==50304 and c['amp_dtype']=='float32'
        assert c['batch_size']*c['grad_accum_steps']*c['block_size']==16384
        assert c['max_tokens']//16384==6103 and c['dropout']==0 and c['bias'] is False
        assert c['optimizer']=='adamw' and c['scalar_lr_mult']==10 and c['peak_lr']==.0006
        assert c['eval_batches']*c['batch_size']*c['block_size']==32768
        if cell['runtime']=='linear':assert c['e062_chunk']==512 and c['e062_validation_seed']==95019
    tasks=[('softmax',[str(ROOT/'checks/verify_softmax.py'),'--source',str(ROOT/'softmax'),'--device','cpu','--out',str(a.out/'softmax.json')]),
           ('linear',[str(ROOT/'checks/verify_linear.py'),'--source',str(ROOT/'linear'),'--device','cpu','--out',str(a.out/'linear.json')]),
           ('strength',[str(ROOT/'checks/verify_strength.py'),'--training-source',str(ROOT/'strength')]),
           ('looped',[str(ROOT/'looped/scripts/verify_looped_hb.py'),'--device','cpu','--out',str(a.out/'looped')])]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for label,argv in tasks:
        with (a.out/(label+'.log')).open('w') as f:subprocess.run([sys.executable,*argv],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180,env=env)
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('shared_targets',ROOT/'aspect/aspect_ratio_protocol.py');m=module_from_spec(spec);spec.loader.exec_module(m)
    assert m.SHAPES==((8,128,8),(16,128,8),(32,128,8),(8,256,8),(8,512,8),(8,128,16),(32,128,16),(8,512,16))
    # The sampler is verified in a fresh process to avoid cross-runtime module caches.
    code="""
import sys,numpy as np,torch
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from data import DataConfig,BlockEpochIterator
from aspect_ratio_protocol import SharedTargetIterator
from dataclasses import replace
cfg=DataConfig(block_size=128,batch_size=8,seed=94020)
data=np.arange(8193,dtype=np.uint16)
for split in ('train','val'):
 batches=[]
 for context in (128,256,512):
  c=replace(cfg,block_size=context,batch_size=1024//context)
  it=SharedTargetIterator(BlockEpochIterator,data,c,split)
  values=[next(it)[1].flatten() for _ in range(16)]
  state=it.state_dict();expected=next(it)
  it.load_state_dict(state);actual=next(it)
  assert all(torch.equal(x,y) for x,y in zip(expected,actual))
  batches.append(torch.cat(values))
 assert all(torch.equal(batches[0],v) for v in batches)
print('PASS shared-macro512 target pairing and state replay across3contexts/2splits')
"""
    with (a.out/'sampler.log').open('w') as f:subprocess.run([sys.executable,'-c',code,str(ROOT/'aspect')],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60,env=env)
    result=dict(status='PASS',presets=len(p['cells']),cpu_formula_gradient_causality_streaming_checks='PASS',shared_target_pairing='PASS',scope='CPU correctness only; no new training or historical GPU/quality guarantee')
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
