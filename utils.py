from nsga2.nsga2.population import Population
import random
from example.model import f1,f2,f3
import pandas as pd

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param

    def create_initial_population(self):
        population = Population()
        count = 0
        while count < self.num_of_individuals:
            individual = self.problem.generate_individual()
            # 计算个体的目标函数值
            features = individual.features
            f1_value = f1(features)  # 使用第一个DNN模型计算目标函数值
            f2_value = f2(features)  # 使用第二个DNN模型计算目标函数值
            f3_value = f3(features)  # 使用第三个DNN模型计算目标函数值
            # 检查目标函数值是否在范围内，如果不在范围内，则舍弃这个个体
            if f1_value >-1 and -1 < f2_value < 0 and -1 < f3_value < 0:
                individual.objectives = [f1_value, f2_value, f3_value]
                population.append(individual)
                count += 1
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            # 计算子代个体的目标函数值
            f1_value_child1 = f1(child1.features)  # 使用第一个DNN模型计算目标函数值
            f2_value_child1 = f2(child1.features)  # 使用第二个DNN模型计算目标函数值
            f3_value_child1 = f3(child1.features)  # 使用第三个DNN模型计算目标函数值
            f1_value_child2 = f1(child2.features)  # 使用第一个DNN模型计算目标函数值
            f2_value_child2 = f2(child2.features)  # 使用第二个DNN模型计算目标函数值
            f3_value_child2 = f3(child2.features)  # 使用第三个DNN模型计算目标函数值
            # 检查目标函数值是否在范围内，如果不在范围内，则舍弃这个子代个体
            if f1_value_child1 >-1 and -1< f2_value_child1 < 0 and -1 < f3_value_child1 < 0:
                child1.objectives = [f1_value_child1, f2_value_child1, f3_value_child1]
                children.append(child1)
            if f1_value_child2 >-1 and -1< f2_value_child2 < 0 and -1< f3_value_child2 < 0:
                child2.objectives = [f1_value_child2, f2_value_child2, f3_value_child2]
                children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        num_of_features = len(child1.features)
        genes_indexes = range(num_of_features)
        for i in genes_indexes:
            beta = self.__get_beta()
            x1 = (individual1.features[i] + individual2.features[i]) / 2
            x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
            child1.features[i] = x1 + beta * x2
            child2.features[i] = x1 - beta * x2
        return child1, child2

    def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))

    def __mutate(self, child):
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u, delta = self.__get_delta()
            if u < 0.5:
                child.features[gene] += delta * (child.features[gene] - self.problem.variables_range[gene][0])
            else:
                child.features[gene] += delta * (self.problem.variables_range[gene][1] - child.features[gene])
            if child.features[gene] < self.problem.variables_range[gene][0]:
                child.features[gene] = self.problem.variables_range[gene][0]
            elif child.features[gene] > self.problem.variables_range[gene][1]:
                child.features[gene] = self.problem.variables_range[gene][1]

    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False



def run_evolution(evo, num_of_generations, num_of_individuals, f1, f2, f3):
    all_data = []

    for i in range(num_of_generations):
        children = evo.utils.create_children(evo.population)
        for child in children:
            f1_value = f1(child.features)
            f2_value = f2(child.features)
            f3_value = f3(child.features)
            child.objectives = [f1_value, f2_value, f3_value]
            all_data.append({
                "Generation": i + 1,
                "Variables": child.features,
                "Objective 1": f1_value,
                "Objective 2": f2_value,
                "Objective 3": f3_value
            })

        evo.population.extend(children)
        evo.utils.fast_nondominated_sort(evo.population)
        new_population = Population()
        front_num = 0
        while len(new_population) + len(evo.population.fronts[front_num]) <= num_of_individuals:
            evo.utils.calculate_crowding_distance(evo.population.fronts[front_num])
            new_population.extend(evo.population.fronts[front_num])
            front_num += 1
        evo.utils.calculate_crowding_distance(evo.population.fronts[front_num])
        evo.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        new_population.extend(evo.population.fronts[front_num][0:num_of_individuals - len(new_population)])
        evo.population = new_population
        evo.utils.fast_nondominated_sort(evo.population)

    return pd.DataFrame(all_data)