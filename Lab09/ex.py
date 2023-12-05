import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import numpy as np


def function():
    df = pd.read_csv(r"C:\Users\enbysenpai\PMP-2023\Lab09\Admission.csv")

    with pm.Model() as model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=2)
        beta2 = pm.Normal('beta2', mu=0, sd=2)

        # modelul regresiei logistice
        pi = pm.math.sigmoid(beta0 + beta1 * df['GRE'] + beta2 * df['GPA'])

        # definim distributia aposteriori
        y = pm.Bernoulli('Admission', p=pi, observed=df['Admission'])

        # sample
        trace = pm.sample(50, tune=50, return_inferencedata=True)


    # granita de decizie si intervalul hdi 94%    
    mean_beta0 = trace.posterior.beta0.mean().values
    mean_beta1 = trace.posterior.beta1.mean().values
    mean_beta2 = trace.posterior.beta2.mean().values

    x1_values = np.linspace(df['GRE'].min(), df['GRE'].max(), 100)
    x2_values = np.linspace(df['GPA'].min(), df['GPA'].max(), 100)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    probabilities = 1 / (1 + np.exp(-(mean_beta0 + mean_beta1 * X1 + mean_beta2 * X2)))
    hdi_94 = az.hdi(probabilities, hdi_prob=0.94)
    print(f"Granita de decizie: {hdi_94}")
    


    # GRE = 550 si GPA = 3.5
    # interval hdi 90% pentru probabilitatea studentului de a fi admis
    gre_new = 550
    gpa_new = 3.5
    new_data = {'GRE': gre_new, 'GPA': gpa_new}

    pi_new = pm.math.sigmoid(mean_beta0 + mean_beta1 * new_data['GRE'] + mean_beta2 * new_data['GPA'])
    hdi_90_new = az.hdi(pi_new, hdi_prob=0.9)

    print(f"HDI 90% sa fie admis studentul cu GRE {gre_new} si GPA {gpa_new}: {hdi_90_new}")


    # GRE = 500 si GPA = 3.2
    # intevral hdi 90% pentru probabilitatea studentului de a fi admis
    gre_new = 500
    gpa_new = 3.2
    new_data = {'GRE': gre_new, 'GPA': gpa_new}

    pi_new = pm.math.sigmoid(mean_beta0 + mean_beta1 * new_data['GRE'] + mean_beta2 * new_data['GPA'])
    hdi_90_new = az.hdi(pi_new, hdi_prob=0.9)

    print(f"HDI 90% sa fie admis studentul cu GRE {gre_new} si GPA {gpa_new}: {hdi_90_new}")

if __name__ == '__main__':
    function()





if __name__ == '__main__':
    function()