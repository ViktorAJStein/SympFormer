"""Factories for exactly the methods and studies in the submitted tables."""
from .core import GPTModel, ModelConfig
from .softmax import NormalizedHBConfig, NormalizedHBModel
from .optim import initialize_learned_v0_tables


def build_model(c):
    method, study = c['method'], c['study']
    if method not in ('B', 'Y', 'H0', 'H1', 'R2'):
        raise ValueError('Method not reported in the paper')
    cfg = ModelConfig(vocab_size=c['vocab'], block_size=c['context'],
                      n_layer=c['layers'], n_head=c['heads'], n_embd=c['width'],
                      dropout=0., bias=False,
                      yurii_force_gain_floor=1e-6 if study in ('linear', 'sigmoid', 'native_l2') or method != 'B' else 0.)
    if study == 'softmax':
        model = GPTModel(cfg) if method == 'B' else NormalizedHBModel(
            cfg, NormalizedHBConfig(method, 'e048_r2_dual_ln_v1' if method == 'R2' else 'e046_identity_v1'))
        if method != 'B':
            initialize_learned_v0_tables(model, c['seed'])
    elif study == 'sharing':
        from .sharing import LayoutDecoder
        model = LayoutDecoder(cfg, method, c['core'], c['seed'])
    elif study == 'linear':
        from .linear import make_model
        model = make_model(cfg, method + '_L', c['seed'], chunk=c['linear_chunk'])
        if method != 'B':
            initialize_learned_v0_tables(model, c['seed'])
    elif study in ('sigmoid', 'native_l2'):
        from .mobility import MobilityConfig, make_model
        family = 'sigmoid_normalized' if study == 'sigmoid' else 'l2_native_tensor_proxy'
        model = make_model(cfg, MobilityConfig(family, method), c['seed'])
        if method != 'B':
            initialize_learned_v0_tables(model, c['seed'])
    else:
        raise ValueError('Study not reported in the submitted paper')
    return model
