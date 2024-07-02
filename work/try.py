import numpy as np
c = np.load("model/cifar10/grad/count.npz")
p = np.load("model/cifar10/grad/percentage.npz")

print(c['epoch40'])
print(p['epoch40'])