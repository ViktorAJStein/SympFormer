# Connection proxies used in the submitted experiments

This document specifies **the executed causal proxies**, not a claim that they
are the Levi–Civita connection of a whole LayerNorm/multi-head language model.
The corresponding implementations are [`softmax.py`](../sympformer/softmax.py),
[`linear.py`](../sympformer/linear.py), and
[`mobility.py`](../sympformer/mobility.py). Numerical definitions are extracted
from the experiment sources; [`SOURCE_PROVENANCE.json`](SOURCE_PROVENANCE.json)
records their original file and definition hashes.

## Shared notation and architectural use

A sequence has N tokens, width d, H heads, head dimension r=d/H. We suppress the
batch index. Token rows z_i and direction rows u_i are written as column vectors
in equations. The base point is `Z = LN_A(X)` for H0/H1/R2 and
`Z = LN_A(X + mu_A V)` for Y. All sums below use **j ≤ i**, including self.

Each head has bias-free learned maps Q_h,K_h,W_h: R^d→R^r, and the concatenated
head result is mapped back by O: R^d→R^d. These are the blocks of `c_attn.weight`
and `c_proj.weight`. There is no score or output symmetry/PSD constraint.

The direction fed to the proxy is the **raw displacement**, not the derivative
of LayerNorm applied to a displacement. In particular, no `D LN(Z)[U]` or
look-ahead tangent map is inserted. Ordinary training autograd differentiates
the implemented expression and its projections normally; this is **not** a
stop-gradient prescription.

Let F be the attention output and let W=beta_A V+gamma_A F, where gamma_A
includes the fixed positive gain floor 10^-6. At each attention substep:

- **B:** native residual attention and native residual MLP; no velocity/proxy.
- **Y:** both attention and MLP look ahead; no correction.
- **H0:** no attention look-ahead, unchanged look-ahead MLP; no correction.
- **H1:** C=Gamma_hat_Z(V,V), V_A=LN_V(W-C), X_A=X+V_A.
- **R2:** C=Gamma_hat_Z(W,W), X_A=X+LN_V(W-C/2), V_A=LN_V(W-C).

Both corrected variants retain
`V_next = LN_V(beta_M V_A + gamma_M MLP(LN_M(X_A + mu_M V_A)))`,
`X_next = X_A + V_next`. The **same learned velocity LN module** is used for
all these normalizations in a block. Turning the correction off gives H0, not B.
The proxy corrects attention only; it is not differentiated through a hypothetical
full attention-plus-MLP geometry. There is no adjustable connection-strength
parameter in these reported studies.

Tied experiments apply two distinct blocks in order 0,1,0,1,0,1,0,1. Velocities
continue through repetitions; they are not reset or reinjected. Untied models
start as eight independent copies of those two initialized blocks. Every logical
application recomputes its own prefix quantities; no sufficient statistic is
carried from one layer/application into another.

## 1. Softmax: identity-form score/value proxy

For head h,

```
q_i = Q_h z_i,  k_j = K_h z_j,  w_j = W_h z_j
ell_ij = q_i^T k_j / sqrt(r)
p_ij = exp(ell_ij) / sum_{m<=i} exp(ell_im)
dot_ell_ij[U] = ((Q_h u_i)^T k_j + q_i^T K_h u_j) / sqrt(r).
```

Define

```
a_i = (1/H) sum_h sum_{j<=i} p_ij dot_ell_ij
s_i = ||u_i||_2^2
F_i = O Concat_h(sum_{j<=i} p_ij w_j)
F_i^(s) = O Concat_h(sum_{j<=i} p_ij s_j w_j).
```

Then

```
Gamma_hat_i(U,U) = a_i u_i - (s_i F_i + F_i^(s))/2.
```

Only the scalar score-direction contraction is averaged over heads; value
aggregations are **concatenated and output-projected**, not averaged. All terms
reuse the same causal softmax probabilities. R2 first computes W from F, then
projects W for the directional contractions, without a second softmax. The
quadratic form in the speed is the identity on full model width, not an inverse
attention Gram matrix. No LayerNorm Hessian or exact multi-head metric is claimed.

## 2. Normalized sigmoid: use the kernel derivative, not softmax weights

Scores and projections are as above, but phi(ell)=sigmoid(ell). With causal sums,

```
D_i = sum_{j<=i} phi(ell_ij)
p_ij = phi(ell_ij)/D_i
E_ij = phi'(ell_ij)/D_i = p_ij sigmoid(-ell_ij).
```

The force is still `F_i = O Concat_h(sum_j p_ij w_j)`. The proxy is

```
a_i = (1/H) sum_h sum_{j<=i} E_ij dot_ell_ij[U]
G_i = O Concat_h(sum_{j<=i} E_ij w_j)
G_i^(s) = O Concat_h(sum_{j<=i} E_ij ||u_j||^2 w_j)
Gamma_hat_i = a_i u_i - (||u_i||^2 G_i + G_i^(s))/2.
```

**G is not F:** using p instead of E in the correction would change the model.
The implementation calculates probabilities as `softmax(logsigmoid(ell))` with
future entries masked to -infinity. The derivative factor is multiplied by
`sigmoid(-ell)`. There is no added sink mass, denominator epsilon, absolute-value
replacement or conversion to signed linear attention in this experiment.

## 3. Signed linear attention: prefix factorwise ridge proxy

This is **not** positive-feature/row-normalized attention. Scale each head's
query and key projections by r^(-1/4):

```
q_j = (Q_h z_j)/r^(1/4),  k_j = (K_h z_j)/r^(1/4)
dot_q_j = (Q_h u_j)/r^(1/4),  dot_k_j = (K_h u_j)/r^(1/4)
f_i^h = (1/i) sum_{j<=i} (q_i^T k_j) w_j.
```

The force is `O Concat_h(f_i^h)`. Scores can be negative. For the inclusive
prefix of length i, let Q,K be i×r row matrices and U the i×d direction matrix.
All five statistics below are **unaveraged prefix sums**. Set lambda=0.001,

```
G_Q = Q^T Q,    G_K = K^T K
P_Q = (G_Q + lambda I_r)^(-1),  P_K = (G_K + lambda I_r)^(-1)
E = P_Q Q^T U                 # r×d
J = P_K K^T dot_K             # r×r
T = P_K K^T Q                 # r×r
L = G_K P_K = I_r - lambda P_K.
```

Writing q_i, dot_q_i and u_i as **row vectors** for this formula, the head's
full-width correction is

```
gamma_i^h = -dot_q_i L E - q_i J^T E
            + u_i E^T T (K_h/r^(1/4))
Gamma_hat_i = (1/H) sum_h gamma_i^h.
```

The last K_h is the r×d **projection matrix**, not the prefix token matrix K.
There is no O or learned value projection in this connection proxy; those maps
still occur in the force. Do not insert them into the correction to make it
resemble the softmax expression. Likewise, do not normalize the Gram sums by i,
move the ridge to a product matrix, or drop L: these change the regularization.

The implementation uses Cholesky solves for the two regularized r×r matrices;
it never forms the inverse of an N×N attention matrix. The five sufficient
statistics are G_Q,G_K,Q^T U,K^T dot_K,K^T Q. Chunk512/checkpoint recomputation
preserves their prefix definitions and is an execution device, not a new metric.
This is a fixed, factorwise-ridge multi-head proxy, not an exact pseudoinverse,
canonical momentum closure, full LayerNorm pullback or the exact unrestricted
bilinear Stein connection.

## 4. Native-L2 attention: positive-definite local tensor proxy

This is the **native force**, not the mean-shift force with w_i subtracted.
Per head,

```
ell_ij = -||Q_h z_i - K_h z_j||^2 / (2 sqrt(r))
p_ij = causal_softmax(ell_ij)
f_i = O Concat_h(sum_{j<=i} p_ij W_h z_j).
```

For full-width f_i, let e_1 be the first coordinate vector and delta=0.001:

```
a_i = softplus(f_i[1]) + delta > 0
t_i = a_i + ||f_i||^2
C_i = I_d + f_i f_i^T/a_i
g_i = C_i^(-1) = I_d - f_i f_i^T/t_i.
```

Both C_i and g_i are positive definite for finite f_i. This is a declared SPD
proxy. **It does not enforce C_i e_1=f_i** and does not require the exact
positive-alignment domain used by a different theoretical native-force tensor.
No sign clipping or manufactured exact-gradient representation is used.

For a direction U, differentiate all projected q,k,w along the **whole prefix**:

```
dot_ell_ij = -((q_i-k_j)^T(dot_q_i-dot_k_j))/sqrt(r)
dot_p_ij = p_ij (dot_ell_ij - sum_m p_im dot_ell_im)
dot_f_i = O Concat_h(sum_j dot_p_ij w_j + p_ij dot_w_j).
```

Hold u_i fixed, and put b_i=f_i^T u_i and a'_i=sigmoid(f_i[1]) e_1. Then

```
dot_t_i = (a'_i + 2 f_i)^T dot_f_i
(D_U g_i) u_i = -(dot_f_i b_i + f_i(dot_f_i^T u_i))/t_i
                + f_i b_i dot_t_i/t_i^2
h_i = grad_{f_i}(u_i^T g_i u_i)
    = -2 b_i u_i/t_i + (b_i/t_i)^2 (a'_i + 2 f_i).
```

Let `self_VJP_i = (D_{z_i} f_i)^T h_i`: **only the derivative with respect to
the current token in its own prefix output**, not the derivative of a global
sum over future query rows. The implemented contraction is

```
v_i = (D_U g_i) u_i - self_VJP_i/2
Gamma_hat_i = C_i v_i = v_i + f_i(f_i^T v_i)/a_i.
```

`native_l2_connection` computes this self-only VJP analytically, including query,
self-key and self-value terms. A global VJP over every row would add suffix
adjoints and violate decoder causality. Consequently this formula is a
**causal local-tensor contraction**, not a claim of the exact Levi–Civita
connection of a single global product metric coupling all tokens. There are no
LayerNorm derivatives in this proxy either. Training autograd remains active
through the analytic calculation.

## Limits and tests

All four proxies are prefix-causal and quadratic in the supplied displacement
at a fixed base point. Those are finite-check targets, not a convergence or
symplecticity theorem. LayerNorm, learned coefficients, head projections and
the retained momentum MLP separate these decoders from the ideal fixed-step
Newton equations. The learned nominal rate `-log(beta)/h` (h=1) is not an exact
post-LayerNorm damping rate; fitted r/t+c curves are descriptive, not a prescribed
Nesterov law. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact study
matrix, controls, sampling/evaluation conventions and hardware limitations.

Run `uv run python -m unittest discover -s tests` for the public CPU checks.
The private publication audit additionally compares output/gradients against
the original frozen numerical definitions; no full GPU training is claimed by
those tests.
