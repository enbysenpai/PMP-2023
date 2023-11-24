# doi jucatori, p0 si p1
# se arunca o moneda normala pentru a decide cine incepe
# jucatorul desemnta arunca cu propria moneda n = {0,1} numarul de steme obtinut
# urmatorul arunca de n+1 ori, fie m numarul de steme obtinut
# p0 este necinstit, cu p(stema) = 1/3
# p1 are moneda normala, cu p(stema)=1/2

# 1. care jucator are sansele mai mari de castig simuland de 20000 ori
# 2. definiti o retea bayesiana
# 3. determinati ce fata a monedei e mai probabil sa fi obtinut in prima runda, stiind ca in a doua nu s-a obtinut nicio stema

import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# simularea unui singur joc
def simulare_joc():
    castig_p0 = 0
    castig_p1 = 0
    # initial se arunca o moneda normala pentru a decide primul jucator
    #       => probabilitate egala pentru a alege oricare dintre cei doi jucatori
    prob_jucator_1 = 1/2
    prob_jucator_2 = 1/2

    nr_simulari = 20000
    for i in range(nr_simulari):
        # alegem in mod random care jucator sa inceapa
        jucator_ales = np.random.choice(['p0','p1'],p=(prob_jucator_1,prob_jucator_2))

        # daca a fost ales jucatorul p0:
        if jucator_ales == 'p0':
            n = 0
            prob_stema = 1/3 # are moneda masluita
            # acest jucator arunca moneda o data si obtine stema cu probabilitate de 1/3
            moneda_p0 = np.random.choice(['stema','ban'],p=(prob_stema, 1-prob_stema))
            if moneda_p0 == 'stema':
                n += 1
            m = 0
            # urmeaza jucatorul p1, care va arunca moneda de n+1 ori
            for i in range(n):
                prob_stema = 1/2
                moneda_p1 = np.random.choice(['stema','ban'],p=(prob_stema,1-prob_stema))
                if moneda_p1 == 'stema':
                    m += 1
            if n < m:
                castig_p1 += 1
            else:
                castig_p0 += 1
        # daca a fost jucatorul p1
        else:
            n = 0
            prob_stema = 1/2 # acesta are moneda normala
            # acest jucator arunca moneda o data si obtine stema cu probabilitate de 1/2
            moneda_p1 = np.random.choice(['stema','ban'],p=(prob_stema, 1-prob_stema))
            if moneda_p1 == 'stema':
                n += 1
            m = 0
            # urmeaza jucatorul p0 care va arunca moneda de n+1 ori
            for i in range(n):
                prob_stema = 1/3
                moneda_p0 = np.random.choice(['stema','ban'],p=(prob_stema,1-prob_stema))
                if moneda_p0 == 'stema':
                    m += 1
            if n < m:
                castig_p0 += 1
            else:
                castig_p1 += 1

    return castig_p0, castig_p1

# definirea retelei Bayesiene
def retea_Bayesiana():
    # se decide jucatorul care va incepe prin aruncarea unei monezi
    # daca incepe p0, probabilitatea sa obtina stema este 1/3, numar steme = n
    # urmeaza p1 sa dea de n+1 ori, cu probabilitatea stemei de 1/2, numar steme = m
    # daca incepe p1, probabilitatea sa obtina stema este de 1/2
    # urmeaza p0
    # daca n >= m, castiga primul jucator
    model_joc = BayesianNetwork([('incepe_p0','n'),('n','m'),('m','castig_p0')])

    cpd_incepe_p0 = TabularCPD(variable='incepe_p0', variable_card=2, values=[[1/2],[1/2]])
    cpd_n = TabularCPD(variable='n', variable_card=2, values=[[2/3,1/2],[1/3,1/2]], evidence=['incepe_p0'], evidence_card=[2])
    cpd_m = TabularCPD(variable='m', variable_card=2, values=[[1/2,2/3],[1/2,1/3]], evidence=['n'], evidence_card=[2])
    cpd_castig_p0 = TabularCPD(variable='castig_p0', variable_card=2, values=[[1,0],[0,1]], evidence=['m'], evidence_card=[2])
    
    model_joc.add_cpds(cpd_incepe_p0, cpd_n, cpd_m, cpd_castig_p0)
    assert model_joc.check_model()

    # reprezentarea grafica a problemei
    pos = nx.circular_layout(model_joc)
    nx.draw(model_joc, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
    plt.show()

    # ce fata a monedei e mai probabil sa se fi obtinut in prima runda stiind ca a doua runda nu s-a obtinut nicio stema
    infer = VariableElimination(model_joc)
    prob_fata_stema02 = infer.query(variables=['n'],evidence={'m':0})
    print(prob_fata_stema02)
    


castig_p0, castig_p1 = simulare_joc()
print(f'P0 castiga de {castig_p0} ori, iar P1 castiga de {castig_p1} ori')

retea_Bayesiana()