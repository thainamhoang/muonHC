"""Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.
Adapted from: https://github.com/KellerJordan/Muon

Usage:
    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'head' not in n]
    other_params = [p for n, p in model.named_parameters() if p not in muon_params]
    optimizer = Muon([
        {'params': muon_params, 'lr': 0.004, 'weight_decay': 0.05},
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.1},
    ])
"""

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to orthogonalize G.
    Uses quintic iteration with coefficients tuned for maximum slope at zero.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.float64 else G
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Normalize spectral norm to at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-Schulz.

    Applies orthogonalization to momentum updates for 2D parameters.
    Should only be used for hidden weight layers. Embeddings, output layers,
    biases, and 1D parameters should use AdamW.

    Args:
        lr: learning rate (spectral norm per update)
        weight_decay: AdamW-style decoupled weight decay
        momentum: momentum coefficient (default 0.95)
        ns_steps: Newton-Schulz iteration steps (default 5)
        nesterov: use Nesterov-style momentum (default True)
    """
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                       ns_steps=ns_steps, nesterov=nesterov)
        params = sorted(params, key=lambda p: p.numel(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - momentum)

                if nesterov:
                    update = grad.lerp(buf, momentum)
                else:
                    update = buf

                # Flatten conv filters to 2D
                original_shape = update.shape
                if update.ndim == 4:  # conv weight
                    update = update.view(update.size(0), -1)

                if update.ndim == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale to preserve approximate norm
                    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                if update.shape != original_shape:
                    update = update.view(original_shape)

                p.add_(update, alpha=-lr)

        return loss