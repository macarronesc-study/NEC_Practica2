import os
import re
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PART 0: EMBEDDED DATASETS
# ==========================================

DATA_FT06 = """
6 6
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
"""

DATA_LA01 = """
10 5
1 21 0 53 4 95 3 55 2 34
0 21 3 52 4 16 2 26 1 71
3 39 4 98 1 42 2 31 0 12
1 77 0 55 4 79 2 66 3 77
0 83 3 34 2 64 1 19 4 37
1 54 2 43 4 79 0 92 3 62
3 69 4 77 1 87 2 87 0 93
2 38 0 60 1 41 3 24 4 83
3 17 1 49 4 25 0 44 2 98
4 77 3 79 2 43 1 75 0 96
"""

DATA_LA29 = """
20 10
8 14 2 38 7 44 0 76 5 97 3 12 4 75 6 66 9 12 1 29
0 43 2 85 3 82 5 38 4 58 9 89 8 92 6 87 7 69 1 80
3 41 7  7 9  5 0 43 2 14 4  8 5 61 1 84 8 66 6 48
2 42 3 74 4 59 6 41 1  8 9 73 8 43 0 96 5 19 7 97
7 23 8 42 4 37 6 55 0  7 5  5 2 70 9 38 3 75 1 48
8  9 6 43 7 31 4 25 5 73 3 95 0 79 2 72 9 60 1 56
1  7 5 21 8 53 6 16 4 94 0 97 3 78 2 64 7 86 9 31
2 65 6 59 7 85 1 33 4 30 8 44 0 61 3 86 9 63 5 32
6 45 2 44 5 61 8 93 1 30 7 90 9 84 4 11 3 16 0 60
4 47 7 36 8 31 1 49 3 20 2 28 6 52 9 35 5 11 0 32
2 77 4 10 9 68 5 17 0 85 1 84 8 20 6 49 7 74 3 34
0 17 5  7 1 85 3 29 2 17 4 76 6 59 8 71 9 13 7 48
6 87 4 39 8 43 7 11 2 15 3 32 5 64 0 19 1 39 9 16
5 33 3 99 6 32 4 91 8 82 2 92 9 99 7 57 1 83 0  8
3 91 5 39 2 69 8 27 7  7 6 21 1 38 9 62 4 88 0 48
2 67 7 80 3 24 0 88 4 18 1 44 8 45 9 64 5 80 6 38
9 59 3 72 6 47 4 40 7 21 5 43 0 51 8 52 1 24 2 15
3 70 2 31 6 20 8 76 1 40 7 43 0 32 5 88 9  5 4 77
4 47 5 64 9 85 3 49 7 58 1 26 0 32 8 80 2 14 6 94
5 59 2 96 0  5 7 79 8 34 4 75 3 26 6  9 9 23 1 11
"""

# ==========================================
# PART 1: DATA STRUCTURES
# ==========================================

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
        lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
        try:
            dims = lines[0].split()
            num_jobs, num_machines = int(dims[0]), int(dims[1])
            jobs_data = []
            for i in range(1, num_jobs + 1):
                row = list(map(int, lines[i].split()))
                job_ops = []
                for j in range(0, len(row), 2):
                    job_ops.append((row[j], row[j+1]))
                jobs_data.append(job_ops)
            return JSSPInstance(name, num_jobs, num_machines, jobs_data)
        except Exception as e:
            return None

    @staticmethod
    def parse_file(filename):
        instances = {}
        if not os.path.exists(filename): return instances
        with open(filename, 'r') as f: lines = f.readlines()
        current_name = None; state = 0; job_count = 0; num_jobs = 0; current_jobs_data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("+++"): continue
            if line.startswith("instance"):
                current_name = line.split()[1]; state = 1; continue
            if state == 1:
                parts = line.split()
                if len(parts) == 2: num_jobs, num_machines = int(parts[0]), int(parts[1]); current_jobs_data = []; job_count = 0; state = 2
                continue
            if state == 2:
                row = list(map(int, line.split()))
                job_ops = [(row[j], row[j+1]) for j in range(0, len(row), 2)]
                current_jobs_data.append(job_ops); job_count += 1
                if job_count == num_jobs:
                    instances[current_name] = JSSPInstance(current_name, num_jobs, num_machines, current_jobs_data); state = 0
        return instances

# ==========================================
# PART 2 & 3: SCHEDULING AND GA
# ==========================================
# (Clases Scheduler y GeneticAlgorithm idénticas al Commit 6, se omiten para brevedad, 
# pero deben estar presentes en el archivo completo)

class Scheduler:
    @staticmethod
    def decode(instance, chromosome, return_schedule=False):
        machine_free_time = [0] * instance.num_machines
        job_next_free_time = [0] * instance.num_jobs
        job_op_index = [0] * instance.num_jobs
        schedule_data = {m: [] for m in range(instance.num_machines)}
        
        for job_id in chromosome:
            op_idx = job_op_index[job_id]
            machine_id, duration = instance.jobs[job_id][op_idx]
            start_time = max(job_next_free_time[job_id], machine_free_time[machine_id])
            end_time = start_time + duration
            machine_free_time[machine_id] = end_time
            job_next_free_time[job_id] = end_time
            job_op_index[job_id] += 1
            if return_schedule: schedule_data[machine_id].append((job_id, start_time, end_time))
            
        makespan = max(machine_free_time)
        return (makespan, schedule_data) if return_schedule else makespan

class GeneticAlgorithm:
    def __init__(self, instance, params):
        self.instance = instance
        self.pop_size = params.get('pop_size', 100)
        self.generations = params.get('generations', 1000)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.elitism_size = max(1, int(self.pop_size * 0.05))
        self.selection_method = params.get('selection', 'tournament')
        self.crossover_method = params.get('crossover', 'JOX')
        self.mutation_method = params.get('mutation', 'swap')
        self.base_genes = []
        for j in range(instance.num_jobs): self.base_genes.extend([j] * instance.num_machines)
        self.population = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = self.base_genes[:]
            random.shuffle(genes)
            self.population.append(genes)

    # ... (Métodos de selección, cruce y mutación iguales al Commit 6) ...

    def select_tournament(self, fitnesses, k=3):
        indices = random.sample(range(self.pop_size), k)
        return self.population[min(indices, key=lambda i: fitnesses[i])]

    def select_roulette(self, fitnesses):
        max_f = max(fitnesses) + 1; probs = [(max_f - f) for f in fitnesses]; total = sum(probs)
        pick = random.uniform(0, total); current = 0
        for i, val in enumerate(probs):
            current += val
            if current > pick: return self.population[i]
        return self.population[-1]

    def crossover_jox(self, p1, p2):
        mask = {j: random.choice([True, False]) for j in range(self.instance.num_jobs)}
        c1 = [-1]*len(p1); c2 = [-1]*len(p2)
        for i, g in enumerate(p1): 
            if mask[g]: c1[i] = g
        for i, g in enumerate(p2): 
            if mask[g]: c2[i] = g
        p2_idx = 0; p1_idx = 0
        for i in range(len(c1)):
            if c1[i] == -1:
                while mask[p2[p2_idx]]: p2_idx += 1
                c1[i] = p2[p2_idx]; p2_idx += 1
        for i in range(len(c2)):
            if c2[i] == -1:
                while mask[p1[p1_idx]]: p1_idx += 1
                c2[i] = p1[p1_idx]; p1_idx += 1
        return c1, c2

    def crossover_pox(self, p1, p2):
        job_set_size = random.randint(1, self.instance.num_jobs - 1)
        job_set = set(random.sample(range(self.instance.num_jobs), job_set_size))
        c1 = [-1]*len(p1); c2 = [-1]*len(p2)
        for i, g in enumerate(p1):
            if g in job_set: c1[i] = g
        for i, g in enumerate(p2):
            if g in job_set: c2[i] = g
        p2_idx = 0; p1_idx = 0
        for i in range(len(c1)):
            if c1[i] == -1:
                while p2[p2_idx] in job_set: p2_idx += 1
                c1[i] = p2[p2_idx]; p2_idx += 1
        for i in range(len(c2)):
            if c2[i] == -1:
                while p1[p1_idx] in job_set: p1_idx += 1
                c2[i] = p1[p1_idx]; p1_idx += 1
        return c1, c2

    def mutate_swap(self, chrom):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(chrom)), 2); chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom

    def mutate_insert(self, chrom):
        if random.random() < self.mutation_rate:
            i = random.randint(0, len(chrom) - 1); gene = chrom.pop(i)
            j = random.randint(0, len(chrom)); chrom.insert(j, gene)
        return chrom

    def run(self):
        self.initialize_population()
        history = []
        best_fitness = float('inf'); best_global = None
        
        for g in range(self.generations):
            fitnesses = [Scheduler.decode(self.instance, ind) for ind in self.population]
            min_fit = min(fitnesses)
            if min_fit < best_fitness:
                best_fitness = min_fit
                best_global = self.population[fitnesses.index(min_fit)][:]
            
            history.append(best_fitness)
            sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])
            new_pop = [self.population[i] for i in sorted_indices[:self.elitism_size]]
            
            while len(new_pop) < self.pop_size:
                p1 = self.select_tournament(fitnesses) if self.selection_method == 'tournament' else self.select_roulette(fitnesses)
                p2 = self.select_tournament(fitnesses) if self.selection_method == 'tournament' else self.select_roulette(fitnesses)
                if random.random() < self.crossover_rate:
                    c1, c2 = self.crossover_jox(p1, p2) if self.crossover_method == 'JOX' else self.crossover_pox(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                c1 = self.mutate_swap(c1) if self.mutation_method == 'swap' else self.mutate_insert(c1)
                c2 = self.mutate_swap(c2) if self.mutation_method == 'swap' else self.mutate_insert(c2)
                new_pop.extend([c1, c2])
            self.population = new_pop[:self.pop_size]
            
        return best_fitness, best_global, history

# ==========================================
# PART 4: SIMULATED ANNEALING
# ==========================================
class SimulatedAnnealing:
    def __init__(self, instance, initial_temp=1000, cooling_rate=0.99):
        self.instance = instance; self.temp = initial_temp; self.cooling = cooling_rate
        self.base = []
        for j in range(instance.num_jobs): self.base.extend([j] * instance.num_machines)
            
    def run(self, max_steps=5000):
        current = self.base[:]; random.shuffle(current)
        current_fit = Scheduler.decode(self.instance, current)
        best = current[:]; best_fit = current_fit; history = []
        
        for _ in range(max_steps):
            if self.temp < 1: break
            neighbor = current[:]; i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            new_fit = Scheduler.decode(self.instance, neighbor)
            
            if new_fit < current_fit:
                current = neighbor; current_fit = new_fit
                if new_fit < best_fit: best_fit = new_fit; best = neighbor
            else:
                if random.random() < math.exp((current_fit - new_fit) / self.temp):
                    current = neighbor; current_fit = new_fit
            history.append(best_fit)
            self.temp *= self.cooling
        return best_fit, best, history

# ==========================================
# PART 5: EXECUTION & PLOTTING
# ==========================================

def plot_gantt(instance, chromosome, title, filename):
    makespan, schedule = Scheduler.decode(instance, chromosome, return_schedule=True)
    colors = plt.cm.tab20(np.linspace(0, 1, instance.num_jobs))
    fig, ax = plt.subplots(figsize=(12, 6))
    for m_id, tasks in schedule.items():
        for job_id, start, end in tasks:
            ax.barh(m_id, end-start, left=start, color=colors[job_id], edgecolor='black', alpha=0.8)
            ax.text((start+end)/2, m_id, f"J{job_id}", ha='center', va='center', fontsize=8, color='white')
    ax.set_title(f'{title} - Makespan: {makespan}'); plt.tight_layout()
    plt.savefig(filename); plt.close()

def main():
    random.seed(51)
    instances = {}
    if os.path.exists('jobshop1.txt'): instances = JSSPLoader.parse_file('jobshop1.txt')
    
    # Cargar datasets embebidos si faltan
    if 'ft06' not in instances: instances['ft06'] = JSSPLoader.load_from_string('ft06', DATA_FT06)
    if 'la01' not in instances: instances['la01'] = JSSPLoader.load_from_string('la01', DATA_LA01)
    if 'la29' not in instances: instances['la29'] = JSSPLoader.load_from_string('la29', DATA_LA29)
    
    target_instances = ['ft06', 'la01', 'la29']
    
    # Configuración inicial para verificar integración
    base_config = {'pop_size': 50, 'generations': 200, 'selection': 'tournament', 'crossover': 'JOX', 'mutation': 'swap'}

    for name in target_instances:
        if name not in instances: continue
        inst = instances[name]
        print(f"\n--- Procesando {name} ---")
        
        # 1. Ejecutar GA
        start = time.time()
        ga = GeneticAlgorithm(inst, base_config)
        ga_fit, ga_chrom, ga_hist = ga.run()
        print(f"GA Best: {ga_fit} (Time: {time.time()-start:.2f}s)")
        
        # 2. Ejecutar SA
        start = time.time()
        sa = SimulatedAnnealing(inst, initial_temp=2000, cooling_rate=0.99)
        sa_fit, sa_chrom, sa_hist = sa.run(max_steps=5000)
        print(f"SA Best: {sa_fit} (Time: {time.time()-start:.2f}s)")
        
        # 3. Graficar Convergencia (Comparativa)
        plt.figure(figsize=(10, 5))
        plt.plot(ga_hist, label=f'GA (Min: {ga_fit})')
        plt.plot(sa_hist, label=f'SA (Min: {sa_fit})', alpha=0.7)
        plt.xlabel('Iteraciones'); plt.ylabel('Makespan')
        plt.title(f'Convergencia: {inst.name}')
        plt.legend(); plt.grid(True)
        plt.savefig(f"{name}_convergence.png")
        plt.close()
        
        # 4. Gantt de la mejor solución global
        if ga_fit < sa_fit:
            plot_gantt(inst, ga_chrom, f"{name} Best (GA)", f"{name}_gantt.png")
        else:
            plot_gantt(inst, sa_chrom, f"{name} Best (SA)", f"{name}_gantt.png")

if __name__ == "__main__":
    main()