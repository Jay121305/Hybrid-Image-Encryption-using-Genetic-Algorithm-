# genetic_algorithm.py
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
from encryption import hybrid_encrypt

def fitness_function(individual, plaintext):
    key1, key2 = individual
    cipher = hybrid_encrypt(plaintext, key1, key2)
    probs = [cipher.count(c)/len(cipher) for c in set(cipher)]
    entropy = -sum([p*np.log2(p) for p in probs]) if len(probs) > 0 else 0.0
    chi_square = sum([(cipher.count(c) - len(cipher)/26)**2/(len(cipher)/26) for c in set(cipher)]) if len(cipher)>0 else 0.0
    return entropy / (1 + chi_square)

def run_ga(plaintext, vig_len=4, perm_len=5, population_size=30, generations=50,
           crossover_rate=0.8, mutation_rate_vig=0.1, mutation_rate_perm=0.2,
           elitism_k=2, verbose=True):

    def random_key1():
        return ''.join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(vig_len))

    def random_key2():
        perm = list(range(perm_len))
        random.shuffle(perm)
        return perm

    population = [(random_key1(), random_key2()) for _ in range(population_size)]
    history_best, history_avg = [], []

    for gen in range(generations):
        fitnesses = [fitness_function(ind, plaintext) for ind in population]
        best_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        if verbose and gen % 5 == 0:
            print(f"Gen {gen:3d} | Best: {best_fit:.4f} | Avg: {avg_fit:.4f}")
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        new_population = sorted_pop[:elitism_k]
        def select():
            total = sum(fitnesses)
            if total == 0:
                return random.choice(population)
            pick = random.uniform(0, total)
            current = 0
            for ind, fit in zip(population, fitnesses):
                current += fit
                if current > pick:
                    return ind
            return population[-1]
        while len(new_population) < population_size:
            p1, p2 = select(), select()
            if random.random() < crossover_rate:
                cut = random.randint(1, vig_len-1) if vig_len>1 else 1
                child1 = (p1[0][:cut] + p2[0][cut:], p1[1][:cut] + p2[1][cut:])
            else:
                child1 = p1
            if random.random() < mutation_rate_vig:
                pos = random.randint(0, vig_len-1)
                s = list(child1[0])
                s[pos] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                child1 = (''.join(s), child1[1])
            if random.random() < mutation_rate_perm:
                a, b = random.sample(range(perm_len), 2)
                child1[1][a], child1[1][b] = child1[1][b], child1[1][a]
            new_population.append(child1)
        population = new_population[:population_size]

    # choose best final individual
    final_fitnesses = [fitness_function(ind, plaintext) for ind in population]
    best_idx = int(np.argmax(final_fitnesses))
    best_ind = population[best_idx]
    # save logs
    os.makedirs("results", exist_ok=True)
    with open("results/ga_fitness_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best", "Average"])
        for i,(b,a) in enumerate(zip(history_best, history_avg)):
            writer.writerow([i,b,a])
    plt.figure()
    plt.plot(history_best, label="best")
    plt.plot(history_avg, label="avg")
    plt.legend(); plt.grid(True)
    plt.savefig("results/ga_fitness_plot.png")
    plt.close()
    return {"best_chromosome": best_ind, "best_fitness": final_fitnesses[best_idx]}
