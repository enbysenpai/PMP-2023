import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az


# timp mediu de asteptare -> distibutie normala de patametri mu si sigma

def solutie():
    # 1. generati 200 de timpi de asteptare folosind distributia de verosimilitate cu parametri alesi de voi
    mu = 10
    sigma = 2
    timpi_asteptare = np.random.normal(mu,sigma,200)

    # 2. descrieti modelul in pymc, folosind ca distributii a priori cele alese mai sus
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=mu, sd=sigma)
        sigma = pm.Normal('sigma',mu=mu,sigma=sigma)
        obs = pm.Normal('obs',mu=mu,sigma=sigma,observed=timpi_asteptare)
        trace = pm.sample(1000,cores=1)
        # cum justificati alegerea facuta? (pentru distributii)
        # alegerea distributiilor a fost facuta in acest mod deoarece am considerat ca acestea se potrivesc cel mai bine cu datele noastre
        # in opinia mea, fiind distributie normala, aceasta are o medie si o deviatie standard care sunt cele mai potrivite pentru a descrie datele noastre

    # 3. estimati folosind modelul de mai sus distributia a posteriori
    #  pentru parametrul sigma
    az.summary(trace)
    az.plot_posterior(trace)
    plt.show()
    




if __name__ == '__main__':
    solutie()