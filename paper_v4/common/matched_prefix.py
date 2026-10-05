"""E045 attention replacements with matched MLPs on persistent prefix ensembles.

State S is a displacement velocity. HB physical U=S/h. This is a NEW
initialization cohort, not an E044 MLP-only intervention or symplectic decoder.
"""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from model import YuriiFormerModel
from riemannian_prefix import PrefixGeometryConfig,PrefixGeometryLayer,RiemannianPrefixModel,DIAGNOSTIC_FIELDS


@dataclass(frozen=True)
class MatchedPrefixConfig:
    attention: str = 'yurii'
    mlp: str = 'yurii'
    version: str = 'e045_displacement_v1'

    def __post_init__(self):
        if self.attention not in ('yurii','hb_fixed','hb_free'):raise ValueError('invalid matched attention')
        if self.mlp not in ('standard','yurii'):raise ValueError('invalid matched MLP')
        if self.version!='e045_displacement_v1':raise ValueError('unsupported matched state/initialization version')


class MatchedPrefixLayer(nn.Module):
    def __init__(self,cfg,geo,matched,donor):
        super().__init__();self.geo=geo;self.matched=matched;self.d=cfg.n_embd
        # Transfer the initialized common modules, rather than reseeding different constructors.
        self.ln_mlp=donor.ln_x_mlp;self.mlp=donor.mlp
        self.ln_v=donor.ln_v if matched.mlp=='yurii' or matched.attention=='yurii' else None
        self.mu_mlp=donor.mu2 if matched.mlp=='yurii' else None
        self.beta_mlp=donor.beta2 if matched.mlp=='yurii' else None
        self.gain_mlp=donor.gamma2 if matched.mlp=='yurii' else None
        self.floor=donor.force_gain_floor
        self.geometry=None
        if matched.attention=='yurii':
            self.ln_attn=donor.ln_x_attn;self.attn=donor.attn
            self.mu_attn=donor.mu1;self.beta_attn=donor.beta1;self.gain_attn=donor.gamma1
            self.work_counts=dict(force_calls=0,connection_calls=0,recompute_force_calls=0,recompute_connection_calls=0)
        else:
            self.geometry=PrefixGeometryLayer(replace(cfg,n_head=1),replace(geo,no_mlp=True))
            self.work_counts=self.geometry.work_counts
        self.recomputing=False;self.capture_diagnostics=False;self.reset_diagnostics()

    def reset_diagnostics(self):
        self.last_geometry={};self._sums={};self._samples=0
        if self.geometry is not None:self.geometry.reset_diagnostics()

    def forward(self,X,S,valid,lengths,layer):
        if self.geometry is not None:
            self.geometry.recomputing=self.recomputing
            self.geometry.capture_diagnostics=self.capture_diagnostics
            Xa,Ua=self.geometry(X,S/self.geo.h,valid,lengths,layer)
            Sa=self.geo.h*Ua
        else:
            key='recompute_force_calls' if self.recomputing else 'force_calls';self.work_counts[key]+=1
            B,P,N,d=X.shape
            look=self.ln_attn(X+self.mu_attn()*S).reshape(B*P,N,d)
            force=self.attn(look).reshape(B,P,N,d)
            Sa=self.ln_v(self.beta_attn()*S+(self.gain_attn()+self.floor)*force)
            Sa=Sa.masked_fill(~valid.unsqueeze(-1),0.);Xa=X+Sa
        if self.matched.mlp=='standard':
            update=self.mlp(self.ln_mlp(Xa));Xnext=Xa+update;Snext=Sa
        else:
            force_mlp=self.mlp(self.ln_mlp(Xa+self.mu_mlp()*Sa))
            Snext=self.ln_v(self.beta_mlp()*Sa+(self.gain_mlp()+self.floor)*force_mlp)
            update=Snext;Xnext=Xa+Snext
        Xnext=Xnext.masked_fill(~valid.unsqueeze(-1),0.);Snext=Snext.masked_fill(~valid.unsqueeze(-1),0.)
        if self.capture_diagnostics and not self.recomputing:
            with torch.no_grad():
                ids=torch.arange(len(lengths),device=X.device)
                norm=lambda z:z[:,ids,lengths-1].norm(dim=-1)
                stats=dict(attention_displacement_norm=norm(Sa),stored_displacement_norm=norm(Snext),mlp_norm=norm(update),total_displacement_norm=norm(Xnext-X))
                if self.geometry is None:stats.update(attention_norm=norm(force),velocity_norm=norm(S),connection_norm=torch.zeros_like(norm(S)),connection_update_norm=torch.zeros_like(norm(S)))
                for k,v in stats.items():self._sums[k]=self._sums.get(k,0.)+float(v.double().sum())
                self._samples+=X.shape[0]*len(lengths)
                self.last_geometry=dict(self.geometry.last_geometry) if self.geometry is not None else {}
                self.last_geometry.update({k:v/self._samples for k,v in self._sums.items()})
        return Xnext,Snext


class MatchedPrefixModel(RiemannianPrefixModel):
    """Reuses only verified prefix distribution/readout/checkpoint machinery."""
    def __init__(self,cfg,geo,matched):
        nn.Module.__init__(self)
        if cfg.n_head!=4 or cfg.dropout!=0:raise ValueError('matched scaffold requires four Yurii heads and dropout0; HB kernel remains single-head')
        expected='free_both' if matched.attention=='hb_free' else 'fixed'
        if geo.solver!='cc' or geo.oracle_norm!='raw' or geo.initial_ln or not geo.learned_v0 or geo.no_mlp or geo.coefficient_policy!=expected:
            raise ValueError('unsupported matched-prefix geometry configuration')
        self.cfg=cfg;self.geo=geo;self.matched=matched
        donor=YuriiFormerModel(cfg,use_v0_init=True)
        self.tok_emb=donor.tok_emb;self.pos_emb=donor.pos_emb
        self.tok_v0_emb=donor.tok_v0_emb;self.pos_v0_emb=donor.pos_v0_emb
        self.layers=nn.ModuleList([MatchedPrefixLayer(cfg,geo,matched,b) for b in donor.blocks])
        self.ln_f=donor.ln_f;self.lm_head=donor.lm_head;self.theta_shared_r=None
        self.last_leak_warnings=0;self.last_h_mean=geo.h if matched.attention!='yurii' else 1.
        self.last_hY_mean=math.nan;self.last_t_start=geo.t0;self.last_t_end=geo.t0+cfg.n_layer*self.last_h_mean
        self.last_c_log_mean=3. if matched.attention=='hb_fixed' else math.nan;self.last_c_lin_mean=0. if matched.attention=='hb_fixed' else math.nan

    def evolve_prefix_batch(self,initial,lengths,initial_velocity=None):
        if initial_velocity is None or initial_velocity.shape!=initial.shape:raise ValueError('matched model requires displacement-velocity initial tables')
        valid=torch.arange(initial.shape[-2],device=initial.device)[None,:]<lengths[:,None]
        X=initial.masked_fill(~valid.unsqueeze(-1),0.);S=initial_velocity.masked_fill(~valid.unsqueeze(-1),0.)
        for k,layer in enumerate(self.layers):X,S=layer(X,S,valid,lengths,k)
        return X,S

    @torch.no_grad()
    def layer_dynamics(self):
        rows=[]
        value=lambda x:float(x) if x is not None else math.nan
        grad=lambda p:float(p.grad.norm()) if p is not None and p.grad is not None else math.nan
        for k,l in enumerate(self.layers):
            y=self.matched.attention=='yurii';h=1. if y else self.geo.h;t=self.geo.t0+k*h
            beta,gain=(l.beta_attn(),l.gain_attn()+l.floor) if y else l.geometry.coefficients(k)
            learned=self.matched.attention in ('yurii','hb_free')
            rows.append(dict(layer=k,shared=0,module_type=type(l).__name__,c_log=3. if not learned else math.nan,c_lin=0. if not learned else math.nan,
                hX=h,hY=math.nan,beta_attn=value(beta),beta_mlp=value(l.beta_mlp() if l.beta_mlp else None),t_start=t,t_end=t+h,
                alpha_start=3/t if not learned else math.nan,alpha_end=3/(t+h) if not learned else math.nan,
                **{name:l.last_geometry.get(name,math.nan) for name in DIAGNOSTIC_FIELDS},
                attention_kind=self.matched.attention,mlp_mode=self.matched.mlp,state_units='displacement; HB physical U=S/h',
                initial_state_version=self.matched.version,effective_heads=4 if y else 1,diagnostic_scope='prefix_readout_particles',
                damping_learned=int(learned),force_gain=value(gain),mlp_force_gain=value(l.gain_mlp()+l.floor if l.gain_mlp else None),
                mlp_lookahead=value(l.mu_mlp() if l.mu_mlp else None),
                attention_displacement_norm=l.last_geometry.get('attention_displacement_norm',math.nan),stored_displacement_norm=l.last_geometry.get('stored_displacement_norm',math.nan),
                token_velocity_gradient=grad(self.tok_v0_emb.weight),position_velocity_gradient=grad(self.pos_v0_emb.weight)))
        return rows
