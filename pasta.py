import types
from contextlib import contextmanager

import torch
import torch.nn.functional as nnfunc

from libs.globals import HyprParams
from libs.model import Model
from libs.splitDataset import BPE

def _headForward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape

    keys = self.key(x)
    queries = self.query(x)

    affinities = queries @ keys.transpose(-2, -1) * C ** -0.5
    affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    affinities = nnfunc.softmax(affinities, dim = -1)

    if self.steer is not None:
        highlight, alpha = self.steer # highlight: (B, T) bool
        scale = torch.where(highlight, affinities.new_tensor(1.0), affinities.new_tensor(alpha)) # shrink non-highlighted columns
        affinities = affinities * scale[:, None, :]
        affinities = affinities / affinities.sum(-1, keepdim = True) # renormalise rows

    affinities = self.dropout(affinities)

    return affinities @ self.value(x)

def install(model) -> None:
    for block in model.blocks:
        for head in block.selfAttention.heads:
            head.steer = None
            head.forward = types.MethodType(_headForward, head)

@contextmanager
def steering(model, layerHeads: dict, mask: torch.Tensor, alpha: float):
    try:
        for layer, heads in layerHeads.items():
            for head in heads:
                model.blocks[layer].selfAttention.heads[head].steer = (mask, alpha)
        yield
    finally:
        for block in model.blocks:
            for head in block.selfAttention.heads:
                head.steer = None

def makeMask(totalLen: int, highlightIdx) -> torch.Tensor:
    crop = min(totalLen, HyprParams.nBlock)
    off = totalLen - crop
    mask = torch.zeros(1, crop, dtype = torch.bool, device = HyprParams.device)
    for p in highlightIdx:
        if p - off >= 0:
            mask[0, p - off] = True

    return mask

def makeRecallBatch(corpus, batch, span = 8, gap = 40, fillerLen = 20):
    T = fillerLen + span + gap + span

    def snippet(n):
        i = torch.randint(0, len(corpus) - n, (1,)).item()
        return corpus[i:i + n]

    xs, highlight, score = [], [], []
    for _ in range(batch):
        S = snippet(span)
        xs.append(torch.cat([snippet(fillerLen), S, snippet(gap), S]))

        h = torch.zeros(T, dtype = torch.bool); h[fillerLen:fillerLen + span] = True
        s = torch.zeros(T, dtype = torch.bool); s[T - span:] = True
        highlight.append(h); score.append(s)

    dev = HyprParams.device
    return torch.stack(xs).to(dev), torch.stack(highlight).to(dev), torch.stack(score).to(dev)

def makeControlBatch(corpus, batch, T = 76, span = 8):
    xs, highlight = [], []
    for _ in range(batch):
        i = torch.randint(0, len(corpus) - T, (1,)).item()
        xs.append(corpus[i:i + T])

        p = torch.randint(0, T - span, (1,)).item()
        h = torch.zeros(T, dtype = torch.bool); h[p:p + span] = True
        highlight.append(h)

    dev = HyprParams.device
    x = torch.stack(xs).to(dev)
    return x, torch.stack(highlight).to(dev), torch.ones_like(x, dtype = torch.bool)

def nll(model, x, scoreMask):
    logits = model(x[:, :-1])
    logp = nnfunc.log_softmax(logits, dim = -1).gather(-1, x[:, 1:, None]).squeeze(-1) # (B, T - 1)
    m = scoreMask[:, 1:]
    return -(logp * m).sum() / m.sum()

@torch.no_grad()
def profileGrid(model, x, highlight, score, alpha):
    mask = highlight[:, :-1] # must match model input length (T - 1)
    base = nll(model, x, score).item()

    grid = torch.zeros(HyprParams.nLayer, HyprParams.nHeads)
    for l in range(HyprParams.nLayer):
        for h in range(HyprParams.nHeads):
            with steering(model, {l: [h]}, mask, alpha):
                grid[l, h] = base - nll(model, x, score).item()

    return grid, base

def selectHeads(recall, control, k, tol):
    ranked = recall.clone()
    ranked[control < -tol] = float('-inf')

    chosen = {}
    for i in torch.topk(ranked.flatten(), k).indices.tolist():
        l, h = divmod(i, recall.shape[1])
        if recall[l, h] > 0:
            chosen.setdefault(l, []).append(h)

    return chosen

def plotGrids(recall, control):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize = (10, 4))
    for ax, g, title in zip(axes, (recall, control), ("recall task (higher = steering helps)", "fluency control (negative = hurts)")):
        im = ax.imshow(g.numpy(), cmap = "coolwarm", vmin = -g.abs().max(), vmax = g.abs().max())
        ax.set_title(title); ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.colorbar(im, ax = ax)

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    model = Model()
    model.load_state_dict(torch.load("./models/ri_gpt_x5_ReLU_2.00.pth", map_location = HyprParams.device, weights_only = True))
    model.to(HyprParams.device)
    install(model)
    model.eval()

    with open("./datasets/data.txt", "r", encoding = "utf-8") as f:
        text = f.read(2000000)
    corpus = torch.tensor(BPE.encode(text))
    print(f"corpus: {len(corpus)} tokens")

    torch.manual_seed(0)
    recall, recallBase = profileGrid(model, *makeRecallBatch(corpus, HyprParams.head_batches), HyprParams.alpha)
    control, controlBase = profileGrid(model, *makeControlBatch(corpus, HyprParams.head_batches), HyprParams.alpha)

    print(f"baseline NLL  recall: {recallBase:.4f}   control: {controlBase:.4f}")
    print("\nrecall grid (rows = layers, cols = heads, positive = steering helps):\n", recall)
    print("\ncontrol grid (negative = steering hurts ordinary text):\n", control)

    heads = selectHeads(recall, control, HyprParams.topk_heads, HyprParams.nll_tolerance)
    print("\nselected heads, paste into HEADS in inference.py:\n", heads)

    torch.save({"recall": recall, "control": control}, "./profile_grids.pt")
    plotGrids(recall, control)
