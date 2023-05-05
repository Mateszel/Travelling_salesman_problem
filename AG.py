import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go


class TSP:
    def __init__(self, city_coords):
        self.n = city_coords.shape[0]
        self.distance_matrix = np.zeros((self.n, self.n))

        for i in range(self.n):
            p1 = city_coords[i]
            for j in range(i + 1, self.n):
                p2 = city_coords[j]

                self.distance_matrix[i, j] = np.linalg.norm(p2 - p1)
                self.distance_matrix[j, i] = np.linalg.norm(p2 - p1)


class Individual:
    def __init__(self, genome):
        self.genome = genome

    def evaluate(self, task):
        cost = 0
        for i in range(len(self.genome) - 1):
            p1 = self.genome[i]
            p2 = self.genome[i + 1]
            cost += task.distance_matrix[p1, p2]

        return cost

    def mutate(self, mutation_rate):
        genome_size = len(self.genome)
        if np.random.rand() < mutation_rate:
            idx1, idx2 = random.choices(range(genome_size), k=2)
            self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]


class Population:
    def __init__(self, genome_size=None, pop_size=None):
        self.size = pop_size
        self.population = []

        if genome_size != None and pop_size != None:
            for _ in range(self.size):
                genome = np.random.permutation(range(genome_size))
                self.population.append(Individual(genome))

    def tournament(self, tournament_size, task):
        selected = random.choices(self.population, k=tournament_size)
        evaluation = [individual.evaluate(task) for individual in selected]
        idx_best_individual = evaluation.index(min(evaluation))
        return selected[idx_best_individual]

    def crossover(self, crossover_rate, parent_1, parent_2, task):
        if np.random.random() < crossover_rate:
            splitting_point = np.random.randint(1, len(parent_1.genome) - 1)
            new_genome = parent_1.genome[:splitting_point]
            new_genome = np.concatenate(
                [new_genome, parent_2.genome[~np.in1d(parent_2.genome, new_genome)]]
            )
            return Individual(new_genome)

        else:
            return parent_1

    def add_child(self, child):
        self.population.append(child)
        self.size = len(self.population)

    def best(self, task):
        evaluation = [individual.evaluate(task) for individual in self.population]
        best_evaluation = min(evaluation)
        best_individual = self.population[evaluation.index(best_evaluation)]
        return best_individual, best_evaluation

    def evaluate(self, task):
        evaluation = np.array(
            [individual.evaluate(task) for individual in self.population]
        )
        return evaluation


class GeneticAlgorithm:
    def __init__(self, populations_size, tournament_size, crossover_rate, mutation_rate):
        self.populations_size = populations_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def fit(self, iterations, task):
        population = Population(genome_size=task.n, pop_size=self.populations_size)
        best_evaluation_history = []
        best_genome_history = []
        mean_evaluation_history = []
        best_individual, best_evaluation = population.best(task)

        for _ in tqdm(range(iterations)):
            new_population = Population()
            for _ in range(population.size):
                parent_1 = population.tournament(self.tournament_size, task)
                parent_2 = population.tournament(self.tournament_size, task)
                child = population.crossover(self.crossover_rate, parent_1, parent_2, task)
                child.mutate(self.mutation_rate)
                new_population.add_child(child)

            best_current_individual, best_current_evaluation = population.best(task)
            best_genome_history.append(best_current_individual.genome)
            if best_current_evaluation < best_evaluation:
                best_individual = best_current_individual
                best_evaluation = best_current_evaluation
            best_evaluation_history.append(best_evaluation)

            mean_evaluation_history.append(population.evaluate(task).mean())

            population = new_population

        return (
            best_individual,
            best_evaluation_history,
            best_genome_history,
            mean_evaluation_history,
        )


def plot_travel(coords, genome, evaluation, file_name, pop, tour, cross, mut, iter):
    df = pd.DataFrame([[coords.loc[idx, 'x'], coords.loc[idx, 'y']] for idx in genome],
                      columns=['y', 'x'])
    title = f'length = {evaluation:.2f}, population:{pop:.0f}, tournament:{tour:.0f}, crossover:{cross:.2f}, mutation:{mut:.1f}, iteration:{iter}'
    fig = px.line(df, x='x', y='y', markers=True, title=title)
    fig.update_layout(
        title_x=0.5
    )
    fig.write_html(f'plot_img/travel_{file_name}.html')


def plot_history(history, file_name):
    df = pd.DataFrame({'x': range(1, len(history) + 1),
                       'y': history})
    fig = px.line(df, x='x', y='y', range_x=[df.x.min() - 5, df.x.max() + 5],
                  labels={'x': 'Iteration', 'y': 'Travel length'},
                  title='Evaluation history')
    fig.update_layout(
        title_x=0.5
    )
    fig.write_html(f'history_img/history_{file_name}.html')


data_path = 'data'
os.listdir(data_path)
dataset = pd.read_csv(os.path.join(data_path, 'dataset_1.csv'), sep=' ', usecols=['x', 'y'])

shortest_line = []
shortest = 100000000

for a in range(2):
    for d in range(4):
        for iter in range(8):
            pop = 50
            tour = 10
            cross = 0.95
            mut = 0.8
            iterations_num = 200
            test_num = d + a*4

            parameters = {'populations_size': pop, 'tournament_size': tour, 'crossover_rate': cross, 'mutation_rate': mut}
            ga = GeneticAlgorithm(**parameters)

            task = TSP(dataset.values)
            best, best_history, best_evaluation, mean_evaluation_history = ga.fit(iterations=iterations_num, task=task)

            # file = f'pop{pop:.0f}_tour{tour:.0f}_cross{cross:.2f}_mut{mut:.1f}'
            file = f'score_{best_history[iterations_num-1]:.0f}_iteration_{test_num+1}_test_{iter+1}'
            plot_travel(dataset, best.genome, best.evaluate(task), file, pop, tour, cross, mut, iter)
            plot_history(best_history, file)

            shortest_line.append(best_history[iterations_num-1])
            if shortest > best_history[iterations_num-1]:
                shortest = best_history[iterations_num-1]
                print(shortest)
                print('Iteration: ', test_num+1, '   Test: ', iter+1)

print(shortest_line)
