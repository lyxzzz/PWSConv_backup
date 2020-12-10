import numpy as np
import math
import matplotlib.pyplot as plt

def f(a, b, c, d):
    print(a)
    print(b)
    print(c)
    print(d)

f(1, 2, 3, 4)

d = {"c":1, "d":2}

f(2, 3, **d)
# x = np.arange(0, 20, 1)
# y1 = []
# y2 = []
# y3 = []
# for t in x:
#     var = 1e-3 - t * 5e-5
#     y_1 = math.sqrt(1 / (var + 5e-3))
#     y1.append(y_1)

#     y_2 = math.sqrt(1 / (var + 1e-3))
#     y2.append(y_2)

#     y_3 = math.sqrt(1 / (var))
#     y3.append(y_3)
# plt.plot(x, y1, label="a")
# plt.plot(x, y2, label="b")
# plt.plot(x, y3, label="b")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.ylim(0, max(max(y1), max(y2), max(y3)))
# plt.legend()
# plt.show()