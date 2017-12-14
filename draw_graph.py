import numpy as np
import matplotlib.pyplot as plt


x = np.array([0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001])
rn = np.array([57.7, 71.3, 71.7, 68.5, 65.7, 63.7])
rt = np.array([74.4, 75.5, 75.7, 75.9, 76.4, 75.0])
rd = np.array([75.2, 80.3, 80.2, 79.0, 79.8, 80.1])
rc = np.array([71.8, 72.4, 74.8, 74.7, 74.6, 74.0])
rs1 = np.array([58.1, 68.2, 72.6, 75.1, 74.5, 72.1])
rs2 = np.array([61.6, 70.2, 74.9, 75.8, 73.3, 67.6])
rs4 = np.array([64.4, 73.8, 74.5, 76.2, 75.7, 70.9])
rs8 = np.array([68.8, 72.2, 71.5, 72.2, 71.2, 70.5])
rs16 = np.array([70.1, 73.3, 71.4, 72.3, 74.1, 69.4])

plt.plot(x, rn, label="RNN")
plt.plot(x, rt, label="RNTN")
plt.plot(x, rd, label="Diag")
plt.plot(x, rc, label="Comp")
plt.plot(x, rs1, label="SMD(m=1)")
plt.plot(x, rs2, label="SMD(m=2)")
plt.plot(x, rs4, label="SMD(m=4)")
plt.plot(x, rs8, label="SMD(m=8)")
plt.plot(x, rs16, label="SMD(m=16)")

plt.xlabel("coefficient of L2 regularization")
plt.ylabel("test 12 accuracy")

plt.legend()
plt.show()
