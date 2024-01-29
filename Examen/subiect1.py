import pandas as pd 
import pymc3 as pm 
import numpy as np 
import matplotlib.pyplot as plt 
import arviz as az 

def encode(data, column):
    mapping_dictionary = {value: index for index, value in enumerate(data[column].unique())}
    data[column] = data[column].map(mapping_dictionary)
    return data

def subiect1():
    df = pd.read_csv(r'C:\Users\enbysenpai\PMP-2023\Examen\Titanic.csv')
    print("Dimensiunea fisierului inainte de a gestiona valorile lipsa =", len(df))
    df = df.dropna()
    print("Dimensiunea fisierului dupa gestionarea valorilor lipsa =", len(df))
    # Am transformat variabila Pclass in variabila numerica:
    df = encode(df, 'Pclass')
    print(df)
    
    
    # definirea modelului pymc3
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 =  pm.Normal('beta2', mu=0, sd=10)

        # definirea modelului de regresie (regresie logistica)
        pi = pm.math.sigmoid(alpha + beta1 * df['Pclass'] + beta2 * df['Age'])

        # definirea distributiei a posteriori
        y = pm.Bernoulli('Survived', p=pi, observed = df['Survived'])

        # sample
        trace = pm.sample(50, tune=50, return_inferencedata = True)

    # calcularea granitei de decizie a datelor
    mean_alpha = trace.posterior.alpha.mean().values
    mean_beta1 = trace.posterior.beta1.mean().values 
    mean_beta2 = trace.posterior.beta2.mean().values 

    x1_values = np.linspace(df['Pclass'].min(),df['Pclass'].max(),100)
    x2_values = np.linspace(df['Age'].min(),df['Age'].max(),100)
    X1,X2 = np.meshgrid(x1_values,x2_values)
    probabilities = 1 /(1 + np.exp(-(mean_alpha + mean_beta1 * X1 + mean_beta2 * X2)))
    hdi_90 = az.hdi(probabilities, hdi_prob=0.9)
    print(f'Granita de decizie: {hdi_90}')


    ''' 
        Dintre varsta si clasa de pasageri, varsta este cea care influenteaza cel mai mult
        daca pasagerul a supravietuit sau nu. Din rezultatele granitei de decizie se observa
        ca valorile luate de variabila x2 asociata varstei este mai mare decat cea asociata
        clasei de calatori. Prin urmare, aceasta va influenta mai mult rezultatul decat ar
        influenta clasa de pasageri.
    '''


    # calculam granita de decizie pentru un pasager cu varsta de 30 de ani aflat in clasa 2a de pasageri
    n_age = 30
    n_pclass = '2a'
    n_data = {'Age':n_age, 'Pclass':n_pclass}
    n_data = encode(n_data,'Pclass')
    pi_new = pm.math.sigmoid(mean_alpha + mean_beta1 * n_data['Pclass'] + mean_beta2 * n_data['Age'])
    hdi_90 = az.hdi(pi_new, hdi_prob=0.9)
    print(f'HDI 90% ca pasagerul cu varsta de {n_age} ani si clasa {n_pclass} este {hdi_90}')


if __name__ == '__main__':
    subiect1()