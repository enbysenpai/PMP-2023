# Price,Speed,HardDrive,Ram,Premium
# model de regresie y ~ N(mu, sigma)
# unde mu = alpha + beta1 * x1 + beta2 * x2
# y - pret de vanzare
# x1 - Speed
# x2 - ln(HardDrive)

import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def lab8():
    # citirea datelor din fisier
    data = pd.read_csv(r"C:\Users\enbysenpai\PMP-2023\Lab08\Prices.csv")
    # nu exista inregistrari ale caror valori sa fie null -> nu necesita curatare

    with pm.Model() as model:
        # calcularea distributiilor apriori
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        # definirea modelului de regresie
        mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])

        # definirea distributiei a posteriori
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])

        # sample
        trace = pm.sample(100, tune=100)
    
    # Vizualizarea distributiei a posteriori
    pm.plot_posterior(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
    plt.show()
    
    # Sumarul distributiei a posteriori
    result = pm.summary(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)

    print(result)

    # sunt Speed si HardDrive predictori utili ai pretului?
    # raspuns: nu, deoarece intervalul de incredere pentru beta1 si beta2 include 0
    #       (intervalul determinat de hdi_2.5% si hdi_97.5%)

    # un consumator este interesat de un computer cu
    # Speed = 33
    # HardDrive = 540
    # simulati 5000 de extrageri din pretul de vanzare asteptat mu
    # construiti un interval de 90% HDI pentru acesta

    with model:
        mu = alpha + beta1 * 33 + beta2 * np.log(540)
        y2 = pm.Normal('y2', mu=mu, sd=sigma, observed=data['Price'])
        trace2 = pm.sample(100, tune=100)
        post_pred = pm.sample_posterior_predictive(trace2, model=model, samples=5000)

    mu = post_pred['y2'].mean(axis=0)
    hdi = pm.stats.hdi(post_pred['y2'], hdi_prob=0.9)
    
    print('HDI:', hdi)
 
    # Vizualizarea distributiei predictive posterioare
    plt.hist(post_pred['y'], bins=30, alpha=0.5, label='Distribuție predictivă posterioară')
    # plt.axvline(data['Price'].mean(), color='red', linestyle='dashed', linewidth=2, label='Media datelor observate')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    lab8()
