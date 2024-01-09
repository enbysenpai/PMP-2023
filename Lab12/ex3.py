import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def metropolis(a_prior, b_prior, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5 
    old_prob = stats.beta.pdf(old_x, a_prior, b_prior)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = stats.beta.pdf(new_x, a_prior, b_prior)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

beta_params = [(1, 1), (20, 20), (1, 4)]

plt.figure(figsize=(10,8))
x = np.linspace(0.01, .99, 100)

for idx, (a_prior, b_prior) in enumerate(beta_params):
    plt.subplot(3, 1, idx+1)
    plt.xlabel('θ')
    
    trace = metropolis(a_prior, b_prior)
    
    prior_distribution = stats.beta(a_prior, b_prior)
    plt.plot(x, prior_distribution.pdf(x), 'r-', label='Distribuție a priori')
    
    plt.hist(trace[trace > 0], bins=25, density=True, label=f'Estimated distribution\n(a_prior={a_prior}, b_prior={b_prior})')
    
    plt.xlim(0, 1)
    plt.ylim(0, 12)
    plt.legend()

plt.tight_layout()
plt.show()


