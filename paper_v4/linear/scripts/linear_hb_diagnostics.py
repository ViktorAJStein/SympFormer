"""Independent per-sequence/layer/place displacement fits and JVP response rates."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scripts.verify_trained_linear_closure import fit


def write_csv(path, rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def collect(model, tokens, out, update):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not hasattr(model,'hb'):
        (out/'diagnostic.json').write_text(json.dumps(dict(mode=model.e062_mode,update=update,damping='not applicable',closure='not applicable'))+'\n')
        return
    previous=[(b.capture_states,b.capture_diagnostics,dict(b.work_counts),b.last_diagnostics,b.last_spectra) for b in model.blocks]
    was_training=model.training;model.eval();rows=[];rates=[];coefficients=[];saved={};spectra={}
    try:
        for b in model.blocks:b.capture_states=b.capture_diagnostics=True
        with torch.no_grad():model(tokens)
        for layer,b in enumerate(model.blocks):
            coeff=model.hb_dynamics()[layer]
            coefficients.append(dict(update=update,**coeff))
            for stage,(x,v) in b.last_states.items():
                for window in range(len(x)):
                    metrics,p=fit(x[window],v[window],return_p=True)
                    rows.append(dict(update=update,mode=model.e062_mode,layer=layer,window=window,stage=stage,**metrics))
                    saved[f'layer{layer}_window{window}_{stage}']=p.numpy()
            if b.last_spectra is not None:spectra[f'layer{layer}']=b.last_spectra.numpy()
            for half,statekey in (('attention','entry'),('mlp','post_attention')):
                x,v=b.last_states[statekey];param=next(b.parameters())
                for window in range(len(x)):
                    xx=x[window:window+1].to(param);vv=v[window:window+1].to(param)
                    tangent=vv if torch.count_nonzero(vv) else torch.ones_like(vv)/math.sqrt(vv.shape[-1])
                    fn=(lambda dv:b.half(xx,dv)[1]) if half=='attention' else (lambda dv:b.mlp_half(xx,dv)[1])
                    with torch.enable_grad():_,jvp=torch.autograd.functional.jvp(fn,vv,tangent,create_graph=False)
                    gain=float(jvp.double().norm()/tangent.double().norm())
                    assert math.isfinite(gain)
                    beta=coeff['beta_attn' if half=='attention' else 'beta_mlp']
                    assert 0<beta<1
                    rates.append(dict(update=update,mode=model.e062_mode,layer=layer,window=window,half=half,
                        beta=beta,gain_force=coeff['gain_attn' if half=='attention' else 'gain_mlp'],
                        forward_step=1.,internal_time=layer+1,nominal_alpha=-math.log(beta),
                        jvp_direction='incoming_displacement' if torch.count_nonzero(vv) else 'fixed_constant_unit_per_token',
                        local_gain=gain,alpha_eff_direction=-math.log(gain) if gain>0 else float('inf'),
                        zero_response=int(gain==0),nesterov_reference=3./(layer+1),
                        scope='local response rate, NOT unique physical damping'))
        np.savez_compressed(out/'p_ls.npz',**saved)
        if spectra:np.savez_compressed(out/'gram_spectra.npz',**spectra)
        write_csv(out/'closure.csv',rows);write_csv(out/'rates.csv',rates)
        write_csv(out/'coefficients.csv',coefficients)
        fits=[]
        for half in ('attention','mlp'):
            for window in range(tokens.shape[0]):
                rs=[r for r in rates if r['half']==half and r['window']==window]
                design=np.array([[1/r['internal_time'],1.] for r in rs]);values=np.array([r['nominal_alpha'] for r in rs])
                rc=np.linalg.lstsq(design,values,rcond=None)[0]
                fits.append(dict(update=update,mode=model.e062_mode,half=half,window=window,
                                 fitted_r=float(rc[0]),fitted_constant=float(rc[1]),
                                 rmse=float(np.sqrt(np.mean((design@rc-values)**2))),learned_law=False))
        write_csv(out/'nesterov_fits.csv',fits)
        (out/'diagnostic.json').write_text(json.dumps(dict(mode=model.e062_mode,update=update,
            fit_unit='sequence/layer/place; never pooled',states=len(rows),rates=len(rates),
            raw_pair='pre-LN algebraic proposal, not executed residual',state_kind='displacement, not canonical momentum'))+'\n')
    finally:
        for b,old in zip(model.blocks,previous):
            b.capture_states,b.capture_diagnostics,b.work_counts,b.last_diagnostics,b.last_spectra=old
            b.last_states={}
        model.train(was_training)
