"""Paper ten-seed study: exactly eight applications, two or eight unique blocks."""
import copy
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .core import GPTModel
from .softmax import NormalizedHBConfig,NormalizedHBModel,attention_connection
from .optim import initialize_learned_v0_tables
from .looped import LoopedDecoder,_norm,_cosine
METHODS=("B","Y","H0","H1","R2")
CORES=(2,8)
VERSION="e074_eight_applications_v1"

class LayoutDecoder(LoopedDecoder):
    def __init__(self,cfg,method,core,seed):
        nn.Module.__init__(self)
        if method not in METHODS or type(core) is not int or core not in CORES:
            raise ValueError('Undeclared E074 method/core')
        if cfg.n_layer!=8 or cfg.dropout!=0 or cfg.bias:
            raise ValueError('Exactly eight applications, dropout0, biasFalse')
        self.cfg=cfg;self.method=method;self.core=core;self.repetitions=8//core
        self.layout='untied' if core==8 else 'tied';self.version=VERSION
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            donor_cfg=replace(cfg,n_layer=2)
            if method=='B':self.net=GPTModel(donor_cfg)
            else:
                version='e048_r2_dual_ln_v1' if method=='R2' else 'e046_identity_v1'
                self.net=NormalizedHBModel(donor_cfg,NormalizedHBConfig(method,version))
                initialize_learned_v0_tables(self.net,seed)
        donor=list(self.net.blocks)
        self.net.blocks=nn.ModuleList([copy.deepcopy(donor[k%2]) for k in range(core)])
        self.net.cfg=cfg;self.last_trace=[]

    def _diagnostic(self,block,x,v,previous_force,j):
        if self.method!='H1':row,force=super()._diagnostic(block,x,v,previous_force,j)
        else:
            z=block.ln_x_attn(x);beta=block.beta1();gain=block.gamma1()+block.force_gain_floor
            force,correction=attention_connection(block.attn,z,v)
            proposal=beta*v+gain*force;off=block.ln_v(proposal);va=block.ln_v(proposal-correction)
            import math
            row=dict(layer=j,t_start=j+1.,t_end=j+2.,forward_step=1.,state_norm=_norm(x),
                damping_applicable=True,beta_attn=float(beta),beta_mlp=float(block.beta2()),
                gain_attn=float(gain),gain_mlp=float(block.gamma2()+block.force_gain_floor),
                mu_attn=0.,mu_mlp=float(block.mu2()),nominal_rate_attn=-math.log(float(beta)),
                nominal_rate_mlp=-math.log(float(block.beta2())),nesterov_reference=3./(j+1.),
                velocity_norm=_norm(v),force_norm=_norm(force),correction_norm=_norm(correction),
                post_ln_velocity_correction_norm=_norm(va-off),post_ln_position_correction_norm=_norm(va-off),
                post_ln_position_cosine=_cosine(va,off)[0],post_ln_position_cosine_valid=_cosine(va,off)[1],
                velocity_force_cosine=_cosine(v,force)[0],velocity_force_cosine_valid=_cosine(v,force)[1],
                successive_force_cosine=None if previous_force is None else _cosine(previous_force,force)[0],
                successive_force_cosine_valid=0 if previous_force is None else _cosine(previous_force,force)[1])
            force=force.detach()
        row.update(loop=j//self.core,core_slot=j%self.core,unique_block=j%self.core)
        return row,force

    def forward(self,idx,targets=None,global_step=None,*,loops=None,capture=False):
        if loops is not None and (type(loops) is not int or loops!=self.repetitions):
            raise ValueError('No alternate-depth training or inference')
        if idx.ndim!=2 or not 0<idx.shape[1]<=self.cfg.block_size:
            raise ValueError('Invalid sequence shape')
        if capture and torch.is_grad_enabled():raise ValueError('Capture requires no_grad')
        pos=torch.arange(idx.shape[1],device=idx.device)
        x=self.net.drop(self.net.tok_emb(idx)+self.net.pos_emb(pos)[None])
        v=None if self.method=='B' else self.net.drop(self.net.tok_v0_emb(idx)+self.net.pos_v0_emb(pos)[None])
        self.last_trace=[];previous=None
        for j in range(8):
            block=self.net.blocks[j%self.core]
            if capture:row,previous=self._diagnostic(block,x,v,previous,j)
            if self.method=='B':x=block(x)
            else:x,v=block(x,v)
            if capture:
                row.update(next_state_norm=_norm(x),next_velocity_norm=None if v is None else _norm(v))
                self.last_trace.append(row)
        logits=self.net.lm_head(self.net.ln_f(x))
        loss=None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten())
        return logits,loss
