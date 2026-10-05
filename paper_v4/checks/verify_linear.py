"""E062 independent dense equations/gradients/null controls/causality/streaming."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F


def setup(source):
    sys.path[:0] = [str(source), str(source/'scripts')]
    from scripts import linear_hb_model as implementation
    from model import ModelConfig
    from train import initialize_learned_v0_tables
    return implementation, ModelConfig, initialize_learned_v0_tables


def make(source, mode, width=8, layers=2, context=16, vocab=31):
    impl, Config, initialize = setup(source)
    cfg = Config(n_embd=width, n_head=2 if width==8 else 4, n_layer=layers,
                 block_size=context, vocab_size=vocab, bias=False, dropout=0.,
                 yurii_force_gain_floor=0. if mode in ('B_S','B_L') else 1e-6)
    m = impl.make_model(cfg, mode, 95019)
    if mode not in ('B_S','B_L'): initialize(m, 95019)
    return m


def reference(attn, z, direction=None):
    b,n,d = z.shape; h,r = attn.n_head, attn.head_dim
    q,k,w = attn.c_attn(z).split(d,-1)
    split = lambda x:x.reshape(b,n,h,r).transpose(1,2)
    q,k,w = split(q)/r**.25,split(k)/r**.25,split(w)
    mask = torch.arange(n,device=z.device)[None,:]<=torch.arange(n,device=z.device)[:,None]
    scores = (q@k.transpose(-1,-2))*mask
    count = torch.arange(1,n+1,device=z.device,dtype=z.dtype)
    force = attn.c_proj((scores@w/count[None,None,:,None]).transpose(1,2).flatten(2))
    if direction is None: return force,None
    dq,dk = F.linear(direction,attn.c_attn.weight[:2*d]).split(d,-1)
    dq,dk = split(dq)/r**.25,split(dk)/r**.25
    wk = attn.c_attn.weight[d:2*d].reshape(h,r,d)/r**.25
    corrections=[]
    for batch in range(b):
        heads=[]
        for head in range(h):
            rows=[]
            for i in range(1,n+1):
                qi,ki,ui = q[batch,head,:i],k[batch,head,:i],direction[batch,:i]
                eye = torch.eye(r,dtype=z.dtype,device=z.device)
                pq = torch.linalg.solve(qi.T@qi+1e-3*eye,eye)
                pk = torch.linalg.solve(ki.T@ki+1e-3*eye,eye)
                inv = ki@pk@pq@qi.T
                dg = dq[batch,head,:i]@ki.T+qi@dk[batch,head,:i].T
                dense = -dg@inv@ui+ui@ui.T@inv.T@qi@wk[head]
                rows.append(dense[-1])
            heads.append(torch.stack(rows))
        corrections.append(torch.stack(heads).mean(0))
    return force,torch.stack(corrections)


def ref_model(m,ids):
    pos=torch.arange(ids.shape[1],device=ids.device)
    x=m.tok_emb(ids)+m.pos_emb(pos)[None]
    v=m.tok_v0_emb(ids)+m.pos_v0_emb(pos)[None]
    for b in m.blocks:
        z=b.ln_x_attn(x+b.mu1()*v) if b.mode in ('Y_S','Y_L') else b.ln_x_attn(x)
        f,_=reference(b.attn,z)
        w=b.beta1()*v+(b.gamma1()+b.force_gain_floor)*f
        _,c=reference(b.attn,z,w if b.mode=='R2_L' else v)
        if b.mode not in ('H1_L','R2_L'): c=c*0
        va=b.ln_v(w-c); xa=x+(b.ln_v(w-.5*c) if b.mode=='R2_L' else va)
        v=b.ln_v(b.beta2()*va+(b.gamma2()+b.force_gain_floor)*b.mlp(b.ln_x_mlp(xa+b.mu2()*va)))
        x=xa+v
    return m.lm_head(m.ln_f(x))


def compare(m,other,ids,ref=None):
    m.zero_grad(set_to_none=True);other.zero_grad(set_to_none=True)
    a=m(ids)[0];b=other(ids)[0] if ref is None else ref(other,ids)
    weights=torch.linspace(.3,1.3,a.numel(),device=a.device,dtype=a.dtype).reshape_as(a)
    (a*weights).sum().backward();(b*weights).sum().backward()
    pa,pb=dict(m.named_parameters()),dict(other.named_parameters());assert pa.keys()==pb.keys()
    errors=[]
    for name in pa:
        ga,gb=pa[name].grad,pb[name].grad
        assert ga is not None and gb is not None and torch.isfinite(ga).all() and torch.isfinite(gb).all(),name
        errors.append(float((ga-gb).norm()/(1+gb.norm())))
    return float((a-b).norm()/(1+b.norm())),max(errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();impl,_,_=setup(source)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids=torch.tensor([[1,3,2,7,8,4],[4,5,6,2,1,3]],device=a.device)
    refs=[];nulls=[];streams=[];causal=[];native=[];checkpoint_checks=[]
    for dtype,tol in ((torch.float64,1e-8),(torch.float32,2e-3)):
        for mode in impl.MODES:
            m=make(source,mode).to(device=a.device,dtype=dtype)
            base=m(ids)[0].detach()
            for cut in range(1,6):
                changed=ids.clone();changed[:,cut:]=(changed[:,cut:]+3)%31
                suffix=float((m(changed)[0][:,:cut]-base[:,:cut]).abs().max())
                cropped=float((m(ids[:,:cut])[0]-base[:,:cut]).abs().max())
                assert suffix<tol and cropped<tol,(mode,suffix,cropped)
                causal.append(dict(mode=mode,dtype=str(dtype),cut=cut,suffix=suffix,crop=cropped))
            if mode not in ('B_S','Y_S'):
                state=None;outputs=[]
                for i in range(ids.shape[1]):
                    value,state=m.stream_step(ids[:,i:i+1],state);outputs.append(value)
                error=float((torch.cat(outputs,1)-base).norm()/(1+base.norm()))
                assert error<tol,(mode,'stream',error)
                assert torch.allclose(m(ids[:1])[0],base[:1],atol=tol,rtol=tol)
                fresh=m.stream_step(ids[:,0:1])[0];assert torch.allclose(fresh,base[:,:1],atol=tol,rtol=tol)
                for token,invalid in ((ids[:,:2],None),(ids[:1,:1],state),(ids[:,:1],dict(state,pos=16)),(ids[:,:1],dict(state,pos=-1))):
                    try:m.stream_step(token,invalid)
                    except ValueError:pass
                    else:raise AssertionError('Invalid stream shape/boundary accepted')
                streams.append(dict(mode=mode,dtype=str(dtype),relative_error=error))
            if mode in ('Y_L','H0_L','H1_L','R2_L'):
                e,g=compare(m,copy.deepcopy(m),ids,ref_model)
                assert max(e,g)<tol,(mode,'reference',e,g)
                refs.append(dict(mode=mode,dtype=str(dtype),output_relative=e,gradient_relative=g))
            if mode in ('H1_L','R2_L'):
                off=make(source,'H0_L').to(device=a.device,dtype=dtype)
                original=impl.linear_proxy
                def zero(*args,**kwargs):
                    c,s,p=original(*args,**kwargs);return c*0,s,p
                with patch.object(impl,'linear_proxy',zero):e,g=compare(m,off,ids)
                assert max(e,g)<tol,(mode,'C0',e,g)
                nulls.append(dict(mode=mode,dtype=str(dtype),output_relative=e,gradient_relative=g))
                other=copy.deepcopy(m)
                from dataclasses import replace
                for block in other.blocks:block.hb=replace(block.hb,checkpoint_chunks=False)
                e,g=compare(m,other,ids);assert max(e,g)<tol
                checkpoint_checks.append(dict(mode=mode,dtype=str(dtype),output_relative=e,gradient_relative=g))
                for scale in (0.,1e-4,1.,100.):
                    torch.manual_seed(95019)
                    x=torch.randn(1,5,8,device=a.device,dtype=dtype,requires_grad=True)
                    v=(scale*torch.randn_like(x)).requires_grad_()
                    out=m.blocks[0](x,v)[0]
                    gx,gv=torch.autograd.grad(out[:,:2].square().sum(),(x,v))
                    assert torch.isfinite(gx).all() and torch.isfinite(gv).all()
                    assert torch.count_nonzero(gx[:,2:])==torch.count_nonzero(gv[:,2:])==0
        from model import YuriiFormerModel
        y=make(source,'Y_S').to(device=a.device,dtype=dtype)
        direct=YuriiFormerModel(y.cfg).to(device=a.device,dtype=dtype);direct.load_state_dict(y.state_dict())
        e,g=compare(y,direct,ids);assert e==g==0;native.append(dict(dtype=str(dtype),Y_S_output=e,Y_S_gradient=g))
    models={mode:make(source,mode) for mode in impl.MODES}
    assert len({sum(p.numel() for p in models[mode].parameters()) for mode in ('H0_L','H1_L','R2_L')})==1
    donor=dict(models['Y_S'].named_parameters())
    for mode,m in models.items():
        for name,value in m.named_parameters():
            mapped=name.replace('.ln_1.','.ln_x_attn.').replace('.ln_2.','.ln_x_mlp.')
            assert torch.equal(value,donor[mapped]),(mode,name)
    from train import build_optimizer
    for mode,m in models.items():
        opt=build_optimizer(m,.0006,optimizer_name='adamw',scalar_lr_mult=10)
        groups={id(p):g for g in opt.param_groups for p in g['params']}
        for name,value in m.named_parameters():
            assert abs(groups[id(value)]['lr']-(.006 if name.endswith('.raw') else .0006))<1e-12
        if hasattr(m,'hb'):assert all(abs(float(b.beta1())-.9)<1e-6 and abs(float(b.gamma1()+b.force_gain_floor)-1)<1e-6 for b in m.blocks)
    result=dict(status='PASS',device=a.device,dense_references=refs,correction_off=nulls,
                streaming=streams,causal_prefixes=causal,native_Y=native,checkpoint_parity=checkpoint_checks,
                common_donor='PASS',optimizer_groups='PASS',scope='correctness only; no quality conclusions')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',a.device,len(refs),'reference gradients',len(nulls),'nulls',len(streams),'streams',len(causal),'prefixes')


if __name__=='__main__':main()
