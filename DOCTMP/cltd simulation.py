import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation parameters
steps = np.arange(1000)  # 1000 steps
T_values = [5, 100, 50]  # Different T values

# Containers for B1 and quality values
B1_curves = {}
quality_curves = {}

# Compute B1 and simulate quality
for T in T_values:
    B1 = np.maximum(5, 100 - steps / T)
    quality = [int(random.uniform(b, 100)) for b in B1]
    B1_curves[T] = B1
    quality_curves[T] = quality

# Plot B1 evolution
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(steps, B1_curves[T], label=f"B1 (T = {T})")
plt.title("Évolution de la borne inférieure B1")
plt.xlabel("step")
plt.ylabel("B1 (min qualité possible)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("simulation_results.png", dpi=300)
# Plot quality evolution
plt.figure(figsize=(12, 6))
for T in T_values:
    plt.plot(steps, quality_curves[T], label=f"quality (T = {T})", alpha=0.7)
plt.title("Évolution des qualités simulées (aléatoires entre B1 et 100)")
plt.xlabel("step")
plt.ylabel("quality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("simulation_results2.png", dpi=300)
# Optional: Plot comparison of B1 and quality for one T (e.g., T = 100)
T_example = 100
plt.figure(figsize=(12, 6))
plt.plot(steps, B1_curves[T_example], label=f"B1 (T = {T_example})", color='blue')
plt.plot(steps, quality_curves[T_example], label=f"quality (T = {T_example})", color='orange', alpha=0.6)
plt.title(f"Comparaison B1 et quality pour T = {T_example}")
plt.xlabel("step")
plt.ylabel("Valeur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("simulation_results3.png", dpi=300)