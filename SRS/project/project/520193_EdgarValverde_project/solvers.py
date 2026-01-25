import numpy as np
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import core_optimization as core

def solve_slsqp(u0, args, limits, history_list):
    print("\n--- Starting SLSQP Optimization ---")

    history_list.clear()

    def constraint_fun(u):
        viol = core.get_constraints_violation(u, args, limits)
        return -viol 

    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    def callback(xk):
        e_pos, e_rot, _ = core.get_detailed_errors(xk, args)
        cost = core.objective_function(xk, args)
        
        record = {
            'iter': len(history_list),
            'err_pos': e_pos,
            'err_rot': e_rot,
            'cost': cost
        }
        history_list.append(record)
        print(f"Iter {len(history_list)} | PosErr: {e_pos:.4f} m | RotErr: {e_rot:.4f}", end='\r')

    res = minimize(core.objective_function, u0, args=(args,), 
                   method='SLSQP', 
                   constraints=cons,
                   options={'maxiter': 150, 'disp': True, 'ftol': 1e-6},
                   callback=callback)
    
    print(f"\nSLSQP Finished. Final Cost: {res.fun:.6f}")
    return res.x

def solve_ga(args, limits, u_bounds, n_gen=100, pop_size=100):
    print("\n--- Starting Genetic Algorithm ---")
    
    dgrs = args[7]
    n_steps = args[8]
    dim = dgrs * n_steps
    
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
        del creator.Individual
        
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, u_bounds[0], u_bounds[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(ind):
        u_arr = np.array(ind)
        cost = core.objective_function(u_arr, args)
        viol = core.get_constraints_violation(u_arr, args, limits)
        total_fitness = cost + (viol * 10000.0) 
        return (total_fitness,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5.0, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=5)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    def get_best_pos_err(population):
        best_ind = tools.selBest(population, 1)[0]
        e_pos, _, _ = core.get_detailed_errors(np.array(best_ind), args)
        return e_pos

    def get_best_rot_err(population):
        best_ind = tools.selBest(population, 1)[0]
        _, e_rot, _ = core.get_detailed_errors(np.array(best_ind), args)
        return e_rot
    
    stats = tools.Statistics()
    stats.register("err_pos", get_best_pos_err)
    stats.register("err_rot", get_best_rot_err)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                       ngen=n_gen, stats=stats, halloffame=hof, 
                                       verbose=True)
    
    best_u = np.array(hof[0])
    
    history = []
    for entry in logbook:
        record = {
            'iter': entry['gen'],
            'err_pos': entry['err_pos'],
            'err_rot': entry['err_rot']
        }
        history.append(record)
        
    return best_u, history