"""Float training of the tiny candidate classifier (PyTorch), then PTQ to int8."""
import numpy as np
import torch
from torch import nn

from .qnn import ARCHS, IN_SCALE, layer_shapes


class Net(nn.Module):
    def __init__(self, arch, win):
        super().__init__()
        self.arch = arch
        self.layers = nn.ModuleList()
        for (typ, co, k, s), (ci, li, _, _) in zip(arch, layer_shapes(arch, win)):
            self.layers.append(nn.Conv1d(ci, co, k, s) if typ == "conv" else nn.Linear(ci, co))

    def forward(self, x, return_acts=False):
        x = x[:, None, :]
        acts = []
        for i, (L, (typ, *_)) in enumerate(zip(self.layers, self.arch)):
            if typ == "dense":
                x = x.flatten(1)
            x = L(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x); acts.append(x)
        return (x[:, 0], acts) if return_acts else x[:, 0]


def train(arch_name, X, y, Xv, yv, win, epochs=30, seed=0, lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    arch = ARCHS[arch_name]
    net = Net(arch, win)
    Xt = torch.tensor(X.astype(np.float32) * IN_SCALE); yt = torch.tensor(y.astype(np.float32))
    Xvt = torch.tensor(Xv.astype(np.float32) * IN_SCALE)
    pw = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad(); loss = lossf(net(Xt[b]), yt[b]); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        _, acts = net(Xt[:50000], return_acts=True)
        act_max = [float(np.percentile(a.numpy(), 99.99)) for a in acts]
        val_logit = net(Xvt).numpy()
    layers = []
    for L in net.layers:
        layers.append((L.weight.detach().numpy().astype(np.float64), L.bias.detach().numpy().astype(np.float64)))
    return net, layers, act_max, val_logit


def float_logits(net, X):
    with torch.no_grad():
        return net(torch.tensor(X.astype(np.float32) * IN_SCALE)).numpy()
