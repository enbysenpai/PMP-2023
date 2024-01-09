import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.repeat([0, 1], (123, 321))
points = 10
h = data.sum()
t = len(data) - h



def posterior_grid1(grid_points = 50, heads = 6, tails = 9):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid1(points, h , t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('theta')
plt.show()



def posterior_grid2(grid_points = 50, heads = 6, tails = 9):
    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid2(points, h , t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('theta')
plt.show()



def posterior_grid3(grid_points = 50, heads = 6, tails = 9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

grid, posterior = posterior_grid3(points, h , t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('theta')
plt.show()