import numpy as np
import matplotlib.pyplot as plt

b = np.genfromtxt("bounds.txt")
b = b[b[:, 0].argsort()]
# x = np.genfromtxt("pdf.txt")
# uf = np.genfromtxt("failed_freqs.txt")

logeps = -10 + np.arange(1000) * 11/999
successful_freqs = []

# for xx in x:
#     f, xx = np.split(xx, [1])
#     successful_freqs.append(f[0])
#     renorm_xx = xx - xx.max()
#     plt.plot(logeps, 10**renorm_xx)

# plt.xlim(-4, 1)
# plt.xlabel("log10eps")
# plt.ylabel("Unnormalized PDF, with max scaled to 1")
# plt.savefig("successful_freqs.png")

# plt.clf()
# plt.hist(successful_freqs, bins=100, label="success", alpha=0.5, density=True)
# plt.hist(uf[:,0], bins=100, label="fail", alpha=0.5, density=True)
# plt.xlabel("frequencies")
# plt.savefig("success_hist.png")

plt.clf()
plt.loglog(b[::10,0], b[::10,1])
plt.xlabel("frequencies")
plt.ylabel("95% CI")
plt.savefig("bounds.png")

