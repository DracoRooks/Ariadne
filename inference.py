import torch

from libs.model import Model
from libs.globals import HyprParams
from libs.splitDataset import BPE
from libs.utils import tokenToString

model = Model()

model.load_state_dict(torch.load("./models/ri_gpt_x4_SwiGLU_1.88.pth", map_location=HyprParams.device, weights_only=True))

print("Begining Inference...")
model.eval()
inputStr = "Once upon a time,"

# out = model.generate(torch.tensor(BPE.encode(inputStr)).view(1, -1).to(HyprParams.device), 1000)
# print(f"output: {tokenToString(BPE.decode(out[0].tolist()))}")

# generate a token individually
inputTensor = torch.tensor(BPE.encode(inputStr)).view(1, -1).to(HyprParams.device)
for i in range(1000):
    out = model.generate(inputTensor, 1)
    outStr = tokenToString(BPE.decode(out[0].tolist()))
    print(outStr, end="", flush=True)

    inputTensor = torch.cat((inputTensor, out), dim = -1)

print("\n\n")
