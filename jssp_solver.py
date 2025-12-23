import os

# Datos de prueba (ft06) para verificar la carga sin fichero externo
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
        self.jobs = jobs_data # Lista de listas de tuplas (maquina, tiempo)

class JSSPLoader:
    @staticmethod
    def load_from_string(name, content):
        """Parsea una cadena raw en una instancia"""
        lines = content.strip().split('\n')
        lines = [l.strip() for l in lines if l.strip()]
        
        try:
            dims = lines[0].split()
            num_jobs = int(dims[0])
            num_machines = int(dims[1])
            
            jobs_data = []
            for i in range(1, num_jobs + 1):
                row = list(map(int, lines[i].split()))
                job_ops = []
                # Formato: maquina tiempo maquina tiempo ...
                for j in range(0, len(row), 2):
                    m = row[j]
                    t = row[j+1]
                    job_ops.append((m, t))
                jobs_data.append(job_ops)
            
            return JSSPInstance(name, num_jobs, num_machines, jobs_data)
        except Exception as e:
            print(f"Error parsing string for {name}: {e}")
            return None

    @staticmethod
    def parse_file(filename):
        """ Parser robusto linea a linea para el formato OR-Lib """
        instances = {}
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return instances

        with open(filename, 'r') as f:
            lines = f.readlines()

        current_name = None
        state = 0 # 0: Look for name, 1: Look for dims, 2: Read Data
        
        job_count = 0
        num_jobs = 0
        num_machines = 0
        current_jobs_data = []

        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith("+++"): continue 

            if line.startswith("instance"):
                current_name = line.split()[1]
                state = 1
                continue

            if state == 1:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    num_jobs = int(parts[0])
                    num_machines = int(parts[1])
                    current_jobs_data = []
                    job_count = 0
                    state = 2
                continue

            if state == 2:
                row = list(map(int, line.split()))
                job_ops = []
                for j in range(0, len(row), 2):
                    m = row[j]
                    t = row[j+1]
                    job_ops.append((m, t))
                current_jobs_data.append(job_ops)
                job_count += 1
                
                if job_count == num_jobs:
                    instances[current_name] = JSSPInstance(current_name, num_jobs, num_machines, current_jobs_data)
                    state = 0 
        
        return instances

if __name__ == "__main__":
    # Test carga desde string
    inst = JSSPLoader.load_from_string('ft06', DATA_FT06)
    if inst:
        print(f"Cargada instancia: {inst.name}, Jobs: {inst.num_jobs}, Maquinas: {inst.num_machines}")
        print("Datos del Job 0:", inst.jobs[0])
    
    # Test carga archivo (si existe)
    if os.path.exists('jobshop1.txt'):
        instances = JSSPLoader.parse_file('jobshop1.txt')
        print(f"Instancias encontradas en archivo: {len(instances)}")