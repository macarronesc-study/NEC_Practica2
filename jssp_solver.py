import os
import random
import time

DATA_FT06 = """
6 6
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
"""

class JSSPInstance:
    def __init__(self, name, num_jobs, num_machines, jobs_data):
        self.name = name
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.jobs = jobs_data

class JSSPLoader:
    @staticmethod
    def load_from_string(name, content):
        lines = content.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        dims = lines[0].split()
        num_jobs = int(dims[0])
        num_machines = int(dims[1])
        jobs_data = []
        for i in range(1, num_jobs + 1):
            row = list(map(int, lines[i].split()))
            job_ops = []
            for j in range(0, len(row), 2):
                job_ops.append((row[j], row[j+1]))
            jobs_data.append(job_ops)
        return JSSPInstance(name, num_jobs, num_machines, jobs_data)

    @staticmethod
    def parse_file(filename):
        instances = {}
        if not os.path.exists(filename): return instances
        with open(filename, 'r') as f: lines = f.readlines()
        current_name = None; state = 0; job_count = 0; num_jobs = 0; num_machines = 0; current_jobs_data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("+++"): continue
            if line.startswith("instance"): current_name = line.split()[1]; state = 1; continue
            if state == 1:
                parts = line.split()
                if len(parts) == 2: num_jobs, num_machines = int(parts[0]), int(parts[1]); current_jobs_data = []; job_count = 0; state = 2
                continue
            if state == 2:
                row = list(map(int, line.split()))
                job_ops = [(row[j], row[j+1]) for j in range(0, len(row), 2)]
                current_jobs_data.append(job_ops); job_count += 1
                if job_count == num_jobs: instances[current_name] = JSSPInstance(current_name, num_jobs, num_machines, current_jobs_data); state = 0
        return instances

class Scheduler:
    @staticmethod
    def decode(instance, chromosome):
        # Tiempo libre de cada maquina
        machine_free_time = [0] * instance.num_machines
        # Tiempo libre de cada job (cuando acaba su operacion anterior)
        job_next_free_time = [0] * instance.num_jobs
        # Indice de la operacion actual para cada job
        job_op_index = [0] * instance.num_jobs
        
        for job_id in chromosome:
            op_idx = job_op_index[job_id]
            machine_id, duration = instance.jobs[job_id][op_idx]
            
            # Puede empezar cuando la maquina este libre Y el job haya acabado lo anterior
            start_time = max(job_next_free_time[job_id], machine_free_time[machine_id])
            end_time = start_time + duration
            
            machine_free_time[machine_id] = end_time
            job_next_free_time[job_id] = end_time
            job_op_index[job_id] += 1
            
        makespan = max(machine_free_time)
        return makespan

class GeneticAlgorithm:
    def __init__(self, instance, params):
        self.instance = instance
        self.pop_size = params.get('pop_size', 100)
        self.generations = params.get('generations', 1000)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.elitism_size = max(1, int(self.pop_size * 0.05))
        
        # Configurable methods
        self.selection_method = params.get('selection', 'tournament')
        self.crossover_method = params.get('crossover', 'JOX')
        self.mutation_method = params.get('mutation', 'swap')
        
        self.base_genes = []
        for j in range(instance.num_jobs):
            self.base_genes.extend([j] * instance.num_machines)
        self.population = []
        self.history = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = self.base_genes[:]
            random.shuffle(genes)
            self.population.append(genes)

    def select_tournament(self, fitnesses, k=3):
        indices = random.sample(range(self.pop_size), k)
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return self.population[best_idx]

    def select_roulette(self, fitnesses):
        max_f = max(fitnesses) + 1
        probs = [(max_f - f) for f in fitnesses]
        total = sum(probs)
        if total == 0: return random.choice(self.population)
        
        pick = random.uniform(0, total)
        current = 0
        for i, val in enumerate(probs):
            current += val
            if current > pick:
                return self.population[i]
        return self.population[-1]

    def crossover_jox(self, p1, p2):
        mask = {j: random.choice([True, False]) for j in range(self.instance.num_jobs)}
        c1 = [-1] * len(p1)
        c2 = [-1] * len(p2)
        
        for i, gene in enumerate(p1):
            if mask[gene]: c1[i] = gene
        for i, gene in enumerate(p2):
            if mask[gene]: c2[i] = gene
            
        p2_idx = 0
        for i in range(len(c1)):
            if c1[i] == -1:
                while mask[p2[p2_idx]]: p2_idx += 1
                c1[i] = p2[p2_idx]
                p2_idx += 1
                
        p1_idx = 0
        for i in range(len(c2)):
            if c2[i] == -1:
                while mask[p1[p1_idx]]: p1_idx += 1
                c2[i] = p1[p1_idx]
                p1_idx += 1
        return c1, c2

    def crossover_pox(self, p1, p2):
        # Precedence Operation Crossover
        job_set_size = random.randint(1, self.instance.num_jobs - 1)
        job_set = set(random.sample(range(self.instance.num_jobs), job_set_size))
        
        c1 = [-1] * len(p1)
        c2 = [-1] * len(p2)
        
        for i, gene in enumerate(p1):
            if gene in job_set: c1[i] = gene
        
        p2_idx = 0
        for i in range(len(c1)):
            if c1[i] == -1:
                while p2[p2_idx] in job_set: p2_idx += 1
                c1[i] = p2[p2_idx]; p2_idx += 1
                
        for i, gene in enumerate(p2):
            if gene in job_set: c2[i] = gene
            
        p1_idx = 0
        for i in range(len(c2)):
            if c2[i] == -1:
                while p1[p1_idx] in job_set: p1_idx += 1
                c2[i] = p1[p1_idx]; p1_idx += 1
        return c1, c2

    def mutate_swap(self, chrom):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(chrom)), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom

    def mutate_insert(self, chrom):
        if random.random() < self.mutation_rate:
            i = random.randint(0, len(chrom) - 1)
            gene = chrom.pop(i)
            j = random.randint(0, len(chrom))
            chrom.insert(j, gene)
        return chrom

    def run(self):
        self.initialize_population()
        self.history = []
        best_fitness = float('inf')
        best_chrom = None
        
        for g in range(self.generations):
            fitnesses = [Scheduler.decode(self.instance, ind) for ind in self.population]
            min_fit = min(fitnesses)
            
            if min_fit < best_fitness:
                best_fitness = min_fit
                best_chrom = self.population[fitnesses.index(min_fit)][:]
            
            self.history.append(best_fitness)
            
            sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
            new_pop = [self.population[i] for i in sorted_indices[:self.elitism_size]]
            
            while len(new_pop) < self.pop_size:
                if self.selection_method == 'tournament':
                    p1 = self.select_tournament(fitnesses)
                    p2 = self.select_tournament(fitnesses)
                else:
                    p1 = self.select_roulette(fitnesses)
                    p2 = self.select_roulette(fitnesses)
                
                if random.random() < self.crossover_rate:
                    if self.crossover_method == 'JOX':
                        c1, c2 = self.crossover_jox(p1, p2)
                    else:
                        c1, c2 = self.crossover_pox(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                
                if self.mutation_method == 'swap':
                    c1 = self.mutate_swap(c1)
                    c2 = self.mutate_swap(c2)
                else:
                    c1 = self.mutate_insert(c1)
                    c2 = self.mutate_insert(c2)
                    
                new_pop.append(c1)
                if len(new_pop) < self.pop_size: new_pop.append(c2)
            
            self.population = new_pop
            
        return best_fitness, best_chrom, self.history

if __name__ == "__main__":
    inst = JSSPLoader.load_from_string('ft06', DATA_FT06)
    # Test new config
    cfg = {'pop_size': 50, 'generations': 100, 'selection': 'roulette', 'crossover': 'POX', 'mutation': 'insert'}
    ga = GeneticAlgorithm(inst, cfg)
    fit, _, _ = ga.run()
    print(f"Resultado con Roulette/POX/Insert: {fit}")