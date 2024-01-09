import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error

N_values = [100, 1000, 10000]
pi_values = []
error_values = []

for N in N_values:
    pi, error = estimate_pi(N)
    pi_values.append(pi)
    error_values.append(error)

mean_error = np.mean(error_values)
std_dev_error = np.std(error_values)

plt.errorbar(N_values, error_values, fmt='o-', label='Error')
plt.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.3f}')
plt.axhline(y=std_dev_error, color='g', linestyle='--', label=f'Standard Deviation of Error: {std_dev_error:.3f}')
plt.xscale('log')  
plt.xlabel('Number of Points (N)')
plt.ylabel('Error (%)')
plt.legend()
plt.show()