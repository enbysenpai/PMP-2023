import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


def f():
    az.style.use('arviz-darkgrid')

    file_path = r'C:\Users\enbysenpai\PMP-2023\Lab10\dummy.csv'

    data = np.loadtxt(file_path)

    x = data[:, 0]
    y = data[:, 1]

    # # 2)
    # num_points = 500
    # x = np.linspace(x.min(), x.max(), num_points)
    # y = np.polyval([2, 1, 0.5], x) + np.random.normal(0, 1, num_points)


    # order = 5
    #3)
    order = 3

    x_1p = np.vstack([x**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)
    y_1s = (y - y.mean()) / y.std()

    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        #a)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        #b) sigma = 100
        # beta = pm.Normal('beta',mu=0,sigma=100,shape=order)
        #b) sigma = [10, 0.1, 0.1, 0.1, 0.1]
        # beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)  # Modificare aici
        epsilon = pm.HalfNormal('epsilon', 5)
        mu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
        idata_p = pm.sample(tune=100,draws=200,return_inferencedata=True)

    
    # Plotare curba
    alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])   
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)


    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    waic_cubic = pm.waic(idata_p)
    loo_cubic = pm.loo(idata_p)
    print("WAIC for cubic model:", waic_cubic.waic)
    print("LOO for cubic model:", loo_cubic.loo)
    
    
    

if __name__ == '__main__':
    f()