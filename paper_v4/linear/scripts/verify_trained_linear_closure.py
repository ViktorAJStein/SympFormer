"""Pure sequence-local LS helper extracted without numerical changes."""
import numpy as np
import torch

def fit(x, y, return_p=False):
    """One sequence/layer/place only; minimum-norm LS when rank deficient."""
    x, y = x.double(), y.double()
    assert x.ndim == y.ndim == 2 and x.shape == y.shape
    assert len(x) > x.shape[1], 'Underdetermined closure fit is not diagnostic'
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    keep = s > 1e-12 * s[0]
    p = (vh[keep].T / s[keep]) @ (u[:, keep].T @ y)
    lam, q = torch.linalg.eigh(x.T @ x)
    rhs = q.T @ (x.T @ y + y.T @ x) @ q
    den = lam[:, None] + lam[None, :]
    valid = den > 1e-12 * den.max()
    ps = q @ torch.where(valid, rhs / den.clamp_min(1e-300), 0.) @ q.T
    yn = float(y.norm())
    rf = float((y-x@p).norm()) / yn if yn else 0.
    rs = float((y-x@ps).norm()) / yn if yn else 0.
    assert rs + 1e-8 >= rf
    normal = x.T @ (x @ p-y)
    assert float(normal.norm()) / (1+float(x.norm()*y.norm())) < 1e-8
    normal_sym = x.T @ (x @ ps-y)
    assert float((normal_sym+normal_sym.T).norm()) / (1+float(x.norm()*y.norm())) < 1e-8
    pn = float(p.norm())
    result = {'r_free': rf, 'r_sym': rs, 'y_norm': yn, 'rank': int(keep.sum()),
              'condition': float(s[0]/s[-1]), 'zero_momentum': int(yn == 0),
              'p_ls_norm': pn, 'p_ls_asymmetry': float((p-p.T).norm())/pn if pn else float('nan'),
              'zero_p_ls': int(pn == 0)}
    return (result, p) if return_p else result

def self_test():
    torch.manual_seed(6136)
    x = torch.randn(13, 4, dtype=torch.float64)
    p = torch.randn(4, 4, dtype=torch.float64); p = (p+p.T)/2
    assert fit(x, x@p)['r_sym'] < 1e-12
    assert fit(x, x@p)['p_ls_asymmetry'] < 1e-12
    p = torch.randn(4, 4, dtype=torch.float64)
    f = fit(x, x@p)
    assert f['r_free'] < 1e-12 and f['r_sym'] > .01
    assert abs(f['p_ls_asymmetry']-float((p-p.T).norm()/p.norm())) < 1e-12
    zero = fit(x, torch.zeros_like(x))
    assert zero['zero_momentum'] == zero['zero_p_ls'] == 1
    assert np.isnan(zero['p_ls_asymmetry'])
    # Representation can fail even when its best-fit matrix is symmetric.
    symmetric = (p+p.T)/2
    noise = torch.randn_like(x)
    noise -= x @ torch.linalg.lstsq(x, noise).solution
    separate = fit(x, x@symmetric+noise)
    assert separate['r_free'] > .01 and separate['p_ls_asymmetry'] < 1e-12
    # Never fit a shared matrix across distinct sequences or layers.
    p2 = p+torch.eye(4, dtype=p.dtype)
    assert fit(x, x@p2)['r_free'] < 1e-12
    assert fit(torch.cat((x, x)), torch.cat((x@p, x@p2)))['r_free'] > .01
    # Nonunique minimum-norm geometry does not exclude symmetric solutions.
    xd = torch.zeros_like(x); xd[:, 0] = x[:, 0]
    ambiguous = fit(xd, xd@symmetric)
    assert ambiguous['r_sym'] < 1e-12 and ambiguous['p_ls_asymmetry'] > .01
    ill = x @ torch.diag(torch.tensor([1., .1, .01, 1e-4], dtype=x.dtype))
    assert fit(ill, ill@p)['r_free'] < 1e-11
    # Independent symmetric design-matrix least squares.
    y = torch.randn_like(x)
    bases = []
    for i in range(4):
        for j in range(i, 4):
            b = torch.zeros(4, 4, dtype=torch.float64)
            b[i, j] = b[j, i] = 1
            bases.append(b)
    design = torch.stack([(x@b).flatten() for b in bases], dim=1)
    c = torch.linalg.lstsq(design, y.flatten()).solution
    residual = float((design@c-y.flatten()).norm()/y.norm())
    assert abs(residual-fit(x, y)['r_sym']) < 1e-12
    xd = x.clone(); xd[:, -1] = xd[:, 0]
    assert fit(xd, xd@p)['r_free'] < 1e-12
    # Randomized independent unrestricted fits across dimensions/conditioning.
    generator = torch.Generator().manual_seed(6137)
    for d in (2, 4, 8):
        for scale in (1., 10., 100.):
            for _ in range(4):
                xr = torch.randn(3*d+1, d, generator=generator, dtype=torch.float64)
                xr *= torch.logspace(0, -np.log10(scale), d, dtype=xr.dtype)
                yr = torch.randn(xr.shape, generator=generator, dtype=xr.dtype)
                metrics, fitted = fit(xr, yr, return_p=True)
                independent = torch.linalg.lstsq(xr, yr, driver='gelsd').solution
                assert torch.allclose(fitted, independent, atol=1e-10, rtol=1e-10)
                asym = float((independent-independent.T).norm()/independent.norm())
                assert abs(metrics['p_ls_asymmetry']-asym) < 1e-10
    # General nonsymmetric unmasked tangent identity (code's row convention).
    a, v = torch.randn(4, 4, dtype=torch.float64), torch.randn(4, 4, dtype=torch.float64)
    y = x@p; s = x.T@x/len(x); alpha = .7
    fx = x@a.T@x.T@y/len(x)
    gy = -y@y.T@x@a.T/len(x)+x@v.T-alpha*y
    dp = v.T-alpha*p-p@p.T@s@a.T-a.T@s@p@p
    assert torch.allclose(gy, fx@p+x@dp, atol=1e-12, rtol=1e-12)
    print('PASS: free/symmetric/asymmetry/zero/rank-deficient/ill-conditioned/separate-fit fixtures; 36 randomized independent LS checks; nonsymmetric global tangent identity')
