import numpy as np 
import arviz as az 
import matplotlib.pyplot as plt 
import pymc3 as pm
from scipy.stats import norm



def f():
    # parametri pentru cele 3 distributii
    clusters = 3
    n_clusters = [200, 150, 100]  # numarul de date pentru fiecare cluster
    n_total = sum(n_clusters)
    means = [5, 0, -5]  # media fiecarui cluster
    std_devs = [2, 2, 2]  # deviatia standard a fiecarui cluster

    mix = np.random.normal(np.repeat(means, n_clusters), np.repeat(std_devs, n_clusters))

    az.plot_kde(np.array(mix))
    plt.hist(mix, bins=50, density=True, alpha=0.5, color="gray")
    plt.show()

    idatas = []
    models = []

    # calibrarea modelelor
    for n_components in range(2,5):
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(n_components))
            means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), n_components), sigma=10, shape=n_components, transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)

            trace = pm.sample(1000, tune=500)
            idata = az.from_pymc3(trace=trace, model=model)
            idatas.append(idata)
            models.append(model)

            # vizualizare rezultate
            plt.figure()
            plt.title(f"Model GMM cu {n_components} componente")
            plt.hist(mix, bins=50, density=True, alpha=0.5, color="gray")

            # pentru a plasa pe acelasi grafic distributiile componente:
            for i in range(n_components):
                mu = idata.posterior["means"].mean(dim=["chain", "draw"])[i]
                sd = idata.posterior["sd"].mean(dim=["chain", "draw"])
                x = np.linspace(mix.min(), mix.max(), 100)
                y = norm.pdf(x, mu, sd)
                plt.plot(x, y, label=f"Componenta {i+1}")

            plt.legend()
            plt.show()

    # comparare modele folosind waic si loo
    comp = az.compare({str(c): idata for c, idata in zip(n_clusters, idatas)},
                 method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(comp)
    plt.show()
    
    comp = az.compare({str(c): idata for c, idata in zip(n_clusters, idatas)},
                 method='BB-pseudo-BMA', ic="loo", scale="deviance")
    az.plot_compare(comp)
    plt.show()



if __name__ == "__main__":
    f()