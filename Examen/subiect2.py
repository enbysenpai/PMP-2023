import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats


def sub2(N):
    x = stats.geom.rvs(0.3,size=(2,N))
    y = stats.geom.rvs(0.5,size=(2,N))
    inside = (x > (y ** y))
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error



N = 1000
print("a) ", sub2(N))

k = 30
pi_values = []
error_values = []

for i in range(k):
    pi, error = sub2(N)
    pi_values.append(pi)
    error_values.append(error)

mean_error = np.mean(error_values)
std_dev_error = np.std(error_values)

print("b) ", mean_error, std_dev_error)
