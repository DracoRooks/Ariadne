import torch

from libs.model import Model
from libs.globals import HyprParams
from libs.splitDataset import BPE
from libs.utils import tokenToString
from pasta import install, steering, makeMask

model = Model()
model.load_state_dict(torch.load("./models/ri_gpt_x5_ReLU_2.00.pth", map_location=HyprParams.device, weights_only=True))
install(model)

print("Begining Inference...")
model.eval()

allHeads = {6: [0, 1, 3], 7: [0, 1], 4: [2]}

# generate a token individually
inputStr = '"Fang Yuan, quietly hand over the Spring Autumn Cicada and I\'ll give you a quick death!"'
inputTensor = torch.tensor(BPE.encode(inputStr)).view(1, -1).to(HyprParams.device)
highlights = [8, 9, 10]
# print(inputTensor.shape[1])

for i in range(512):
    mask = makeMask(inputTensor.shape[1], highlights)

    with steering(model, allHeads, mask, 0.01):
        out = model.generate(inputTensor, 1)

    outStr = tokenToString(BPE.decode(out[0].tolist()))
    print(outStr, end="", flush=True)

    inputTensor = torch.cat((inputTensor, out), dim = -1)
