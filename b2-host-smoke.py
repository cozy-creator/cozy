import sys, tempfile
from pathlib import Path
import torch
from torch import nn
import tensorfs
from tensorfs.varena.torch import Declared, Level, Memory, WeightGrant, blocks_of, regions_of

class Block(nn.Module):
    def __init__(self, d): super().__init__(); self.norm = nn.LayerNorm(d); self.lin = nn.Linear(d, d)
    def forward(self, x): return x + self.lin(self.norm(x))

class Net(nn.Module):
    _no_split_modules = ["Block"]
    def __init__(self, d, n): super().__init__(); self.embed = nn.Linear(d, d); self.blocks = nn.ModuleList(Block(d) for _ in range(n)); self.head = nn.Linear(d, d)
    def forward(self, x):
        x = self.embed(x)
        for b in self.blocks: x = b(x)
        return self.head(x)

torch.manual_seed(0)
ref = Net(64, 4)
root = Path(tempfile.mkdtemp()) / "store"
store = tensorfs.Store(str(root))
w = store.writer()
for k, t in ref.state_dict().items():
    t = t.detach().contiguous()
    w.put("net", k, "F32", list(t.shape), t.numpy().tobytes())
ck = store.checkpoint(w.commit())
with torch.device("meta"):
    net = Net(64, 4)
net._varena = Declared(WeightGrant(ck, "net", "sha256:" + "0" * 64))
print("blocks", blocks_of(net)); print("regions", len(regions_of(net)))
mem = Memory.open(torch.device("cpu"))
model = mem.load(net)
assert mem.load(net) is model and mem.sealed() == []
x = torch.randn(3, 64)
with torch.no_grad():
    want = ref(x)
    with mem.request("r1") as r:
        got = net(x)
assert torch.equal(got, want), (got - want).abs().max()
s = mem.stats()
print("stats", [(m.name, m.resident, m.level) for m in s.models], r.stats)
mem.load(net, "ram"); assert model.level is Level.RAM
model.close()
print("OK")
