import matplotlib.pyplot as plt
import json
import numpy as np

run = "runs/2022-10-19 19:25:29.889364"
with open(f"{run}/run_stats.json") as f:
    dat = np.array(json.load(f)["losses_G_D"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dat)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig("plot.png")
