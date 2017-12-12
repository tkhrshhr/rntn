import numpy as np
import matplotlib.pyplot as plt


x = np.array([0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.001])
rn = np.array([])
rt = np.array([])
rd = np.array([])
rc = np.array([])
rs1 = np.array([])
rs2 = np.array([])
rs4 = np.array([])
rs8 = np.array([])
rs16 = np.array([])

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
plt.ylabel("validation accuracy")
plt.title('Regularization Sensitivity')

plt.legend()
plt.show()
