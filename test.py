import matplotlib.pyplot as plt
import torch
import torchsde
from multistablesde.models.triple_well import TripleWell
from multistablesde.models.energy_balance_constant import ConstantStochasticEnergyBalance
import sys
sys.setrecursionlimit(10000)

model = TripleWell()
# model = ConstantStochasticEnergyBalance()

T = torch.linspace(-1, 1, 1000)
energy_diff = model.drift(T)
plt.plot(T, energy_diff)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Temperature (K)")
plt.ylabel("Net Energy Flux")
plt.savefig("test.pdf")

# plt.cla()

# device = torch.device("cpu")
# model = TripleWell()
# ts = torch.linspace(0, 50, 100).to(device)
# batch_size = 1
# xs, mean, std = model.sample(batch_size, ts, True, device)

# # Plot first trajectory
# plt.plot(ts.cpu().numpy(), xs[:, 0, 0].cpu().numpy())
# plt.xlabel("Time")
# plt.ylabel("Normalized x(t)")
# plt.title("Triple-Well SDE Sample (1 of 16)")
# plt.grid(True)
# plt.savefig("sample_path.pdf")
