from shutil import Error
import cplex
import random
import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import time

@dataclass()
class CplexModel:
    rr_filename: str
    cov_filename: str
    expect_return: float
    c: cplex.Cplex = cplex.Cplex()


    def __post_init__(self):
        # init data
        self.cov_matrix = pd.read_csv(self.cov_filename).to_numpy()[:,1:]
        self.rr_matrix = pd.read_csv(self.rr_filename, index_col=0) # load data
        
        # init size
        self.size = self.rr_matrix.shape[0]
        self.names = [f"w{i}" for i in range(self.size)]
    
    def run_quad_solver(self):
        self.c.variables.add(obj=[0]*self.size, lb=[0]*self.size, ub=[1]*self.size, names=self.names) # geneate vairables (weights)
        
        rows = [[self.names, [1]*self.size], #sum should be one 
                [self.names, self.rr_matrix["return"].values.tolist()]] # return rate
        
        self.c.linear_constraints.add(lin_expr=rows, senses='EG', rhs = [1, self.expect_return])
        
        #quadratic part:
        qmat = []
        
        for j in range(self.size):
            qmat.append([[i for i in range(self.size)], [2*self.cov_matrix[j, m] for m in range(self.size)]]) 
        
        self.c.objective.set_quadratic(qmat)

        self.c.solve()
    
    def generate_readable_solution(self):
        selected_stock = []
        selected_weights = []
        for x in range(self.size):
            val = np.round(self.c.solution.get_values()[x], 3) # only values that are 10 ^-3
            if val != 0:
                selected_stock.append(self.rr_matrix.index.values[x])
                selected_weights.append(val)
        df = pd.DataFrame({"stock": selected_stock, "weight": selected_weights})
        print('Selected stocks and their weights:')
        print(df)
        print('\nPortfolio risk:')
        print(round(self.c.solution.get_objective_value(),7))
        # print(sum(df["weight"].values.tolist()))

@dataclass
class SimAnneling:
    
    rr_filename: str
    cov_filename: str
    expect_return: float

    def __post_init__(self):
        # init data
        self.cov_matrix = pd.read_csv(self.cov_filename).to_numpy()[:,1:]
        self.rr_matrix = pd.read_csv(self.rr_filename, index_col=0) # load data
        self.return_val = np.array(self.rr_matrix["return"].values.tolist())
        
        # init size
        self.size = self.rr_matrix.shape[0]
        self.names = [f"w{i}" for i in range(self.size)]
        self.state = np.array([1/self.size]*self.size)
        self.curr_best = 1000000
        self.best_w = np.array([1/self.size]*self.size)

    def eval(self, new = None):
        if new is not None:
            risk = np.linalg.multi_dot([new, self.cov_matrix, new])/2
        else: 
            risk = np.linalg.multi_dot([self.state, self.cov_matrix, self.state])/2
        
        return risk
    
    def gen_valid_w(self):
        
        new_w = self.best_w
        cnt = 0
        change_to_apply = min(0.5, float(self.temp/10))
        for i in range(self.size):
            new_w[i] = new_w[i] * random.choice([change_to_apply, 1+change_to_apply])
        
        new_w = new_w / sum(new_w)
        
        while (new_w @ self.return_val < self.expect_return) or (sum(new_w).__round__(4) != 1):
            for i in range(self.size):
                new_w[i] = new_w[i] * random.choice([change_to_apply, 1+change_to_apply])
        
            new_w = new_w / sum(new_w)
            # print(f"Iter: {cnt}| Return {new_w @ self.return_val}| Sum: {sum(new_w).__round__(2)} | Size: {change_to_apply}| Temp:{self.temp}")
            cnt += 1
            if cnt == 2000:
                new_w = np.random.rand(self.size)
                new_w = new_w / sum(new_w)
                cnt = 0 
 

        return new_w

    def run_sim(self, init_temp = 100, final_temp = 0.1, alpha=0.05):
        debug = False
        states = []
        costs = []
        self.temp = init_temp

        """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
        
        cost = self.eval()

        while self.temp > final_temp:
            new_state = self.gen_valid_w()
            new_cost = self.eval(new_state)
            if debug: 
                print(f"{self.temp} | Temp current: {float(self.temp).__round__(7)} | New {float(new_cost).__round__(7)}")
            if (new_cost < cost) or (random.random() < (math.e ** (-(new_cost-cost)/self.temp))):
                self.state, cost = new_state, new_cost
                states.append(self.state)
                costs.append(cost)
            self.temp -= alpha

        # return self.state, self.eval(), states, costs

    def generate_readable_solution(self):
        selected_stock = []
        selected_weights = []
        for x in range(self.size):
            val = np.round(self.state[x], 3) # only values that are 10 ^-3
            if val != 0:
                selected_stock.append(self.rr_matrix.index.values[x])
                selected_weights.append(val)
        df = pd.DataFrame({"stock": selected_stock, "weight": selected_weights})
        print('Selected stocks and their weights:')
        print(df)
        print('\nPortfolio risk:')
        print(round(self.eval(), 7))
        # print(sum(df["weight"].values.tolist()))


if __name__ == "__main__":
    c = CplexModel('43best.csv', "43StocksCovariance.csv", 0.10)
    c.run_quad_solver()
    c.generate_readable_solution()
    s = SimAnneling('43best.csv', "43StocksCovariance.csv", 0.08)
    st = time.time()
    data = s.run_sim()
    end = time.time()
    s.generate_readable_solution()
    print(f"Metaheuristic time: {(end-st).__round__(2)} seconds")
