import arviz as az 
import matplotlib.pyplot as plt  


centered_eight_data = az.load_arviz_data("centered_eight")
non_centered_eight_data = az.load_arviz_data("non_centered_eight")


"""

    1) Pentru fiecare model, identificati numarul de lanturi, 
       marimea totala a esantionului generat
       + vizualizare distributie a posteriori

"""


# Modelul centrat
print("Detalii model centrat: ")
print("Numarul de lanturi: ", centered_eight_data.posterior.chain.size)
print("Marime esantion: ", centered_eight_data.posterior.draw.size)
# Vizualizarea distributiei a posteriori
az.plot_posterior(centered_eight_data) 
plt.show()


print("\n")
#Modelul necentrat
print("Detalii modelul necentrat: ")
print("Numarul de lanturi: ", non_centered_eight_data.posterior.chain.size)
print("Marime esantion: ", non_centered_eight_data.posterior.draw.size)
# Vizualizarea distributiei a posteriori
az.plot_posterior(non_centered_eight_data)
plt.show()


print("\n")
"""

    2) Comparati cele doua modele, dupa criteriile Rhat si autocorelatie.

"""


print("Compararea modelelor: ")
print("Modelul centrat: ")
print(f"Criteriul Rhat: mu = {az.rhat(centered_eight_data.posterior['mu'].values)}; tau = {az.rhat(centered_eight_data.posterior['tau'].values)}")
print(f"Autocorelatie: mu = {az.autocorr(centered_eight_data.posterior['mu'].values)}; \ntau = {az.autocorr(centered_eight_data.posterior['tau'].values)}")
az.plot_autocorr(centered_eight_data)
plt.show()
print("\n")
print("Modelul necentrat: ")
print(f"Criteriul Rhat: mu = {az.rhat(non_centered_eight_data.posterior['mu'].values)}; \ntau = {az.rhat(non_centered_eight_data.posterior['tau'].values)}")
print(f"Autocorelatie: mu = {az.autocorr(non_centered_eight_data.posterior['mu'].values)}; \ntau = {az.autocorr(non_centered_eight_data.posterior['tau'].values)}")
az.plot_autocorr(non_centered_eight_data)
plt.show()


print("\n")
"""

    3) Numarati numarul de divergente din fiecare model, iar apoi identificati
       unde tind sa se concentreze in spatiul parametrilor

"""

print("Modelul centrat: ")
divergences_centered = centered_eight_data.sample_stats.diverging.sum()
print("Numar divergente: ", divergences_centered)
az.plot_pair(centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.show()
print("\n")
print("Modelul necentrat: ")
divergences_non_centered = non_centered_eight_data.sample_stats.diverging.sum()
print("Numar de divergente: ", divergences_non_centered)
az.plot_pair(non_centered_eight_data, var_names=["mu", "tau"], divergences=True)
plt.show()