import random

# Example Vigenère keyspace (A–Z only)
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def fitness_function(candidate):
    """Dummy fitness — you can plug your own logic here."""
    # Just a placeholder function simulating fitness scores
    return sum(ord(c) for c in candidate) % 100 / 100.0

def run_genetic_algorithm(
    population_size=20,
    generations=40,
    key_length=4,
    image_key_length=32
):
    """Runs a simple GA to evolve text and image keys."""
    print("Running GA — this may take a bit...")

    # Initialize population
    population = [''.join(random.choice(CHARSET) for _ in range(key_length))
                  for _ in range(population_size)]

    best_candidate = None
    best_fitness = 0.0

    for gen in range(generations + 1):
        fitness_scores = [fitness_function(ind) for ind in population]

        # Track best
        best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_candidate = population[best_idx]

        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        if gen % 5 == 0:
            print(f"Gen {gen:3d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

        # Selection: pick top 50%
        selected = [population[i] for i in sorted(range(len(fitness_scores)),
                    key=lambda i: fitness_scores[i], reverse=True)[:population_size // 2]]

        # Crossover + Mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            cut = random.randint(1, key_length - 1)
            child = parent1[:cut] + parent2[cut:]
            if random.random() < 0.2:
                pos = random.randint(0, key_length - 1)
                child = child[:pos] + random.choice(CHARSET) + child[pos + 1:]
            new_population.append(child)

        population = new_population

    # Convert text key to numeric image key (dummy mapping)
    image_key = [random.randint(0, 31) for _ in range(image_key_length)]

    return best_candidate, image_key


# Allow testing directly
if __name__ == "__main__":
    text_key, image_key = run_genetic_algorithm()
    print(f"Text key: {text_key}")
    print(f"Image key: {image_key}")
