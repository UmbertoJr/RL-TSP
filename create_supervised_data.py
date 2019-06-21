from tqdm import tqdm
import numpy as np
from dataGenerator import DataGenerator
from concorde.tsp import TSPSolver
import os
from configuration import get_config
from multiprocessing import Pool
import pickle
import sys
import os


def pool_operation(input_b, it):
    from concorde.tsp import TSPSolver
    import os
    # Concorde
    solver = TSPSolver.from_data(
        input_b[it, :, 0] * 1000,
        input_b[it, :, 1] * 1000,
        norm="EUC_2D"
    )
    # Find tour
    solution = solver.solve()
    return solution.tour


config = get_config()

dataset = DataGenerator()
real_tour_concorde = {}
pool = Pool(processes=8)
last = 10000
if not os.path.exists("supervised_data"):
    os.mkdir("supervised_data")

for i in tqdm(range(10000, 30000)):  # test instance
    seed_ = 1 + i
    reward, tour = [], []
    input_batch, dist_batch = dataset.create_batch(config.batch_size, config.graph_dimension,
                                                   config.dimension, seed=seed_)

    sys.stdout = open(os.devnull, "w")
    result_tours = [pool.apply(pool_operation, args=(input_batch, x)) for x in range(config.batch_size)]
    sys.stdout = sys.__stdout__

    real_tour_concorde[seed_] = result_tours

    if seed_ % 100 == 0:
        f = open("supervised_data/batch_seed_" + str(last) + "_" + str(seed_) + ".pkl", "wb")
        pickle.dump(real_tour_concorde, f)
        f.close()
        print("saved batch from seed " + str(last) + " to seed " + str(seed_))
        real_tour_concorde = {}
        last = seed_

print("Creation COMPLETED !")

# np.savetxt("risultati/concorde_len_TSP"+str(actor.max_length)+".txt", real_len_concorde)
