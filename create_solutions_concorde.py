from tqdm import tqdm
import numpy as np
from dataGenerator import DataGenerator
from configuration import get_config
from concorde.tsp import TSPSolver


config = get_config()

dataset = DataGenerator()
real_tour_concorde = {}

real_len_concorde = []
for i in tqdm(range(1000)):  # test instance
    seed_ = 1 + i
    input_batch, dist_batch = dataset.create_batch(1, config.graph_dimension,
                                                   config.dimension, seed=seed_)

    # Concorde
    solver = TSPSolver.from_data(
        input_batch[0, :, 0]*1000,
        input_batch[0, :, 1]*1000,
        norm="EUC_2D"
    )
    # Find tour
    solution = solver.solve()
    real_len_concorde.append(solution.optimal_value/1000)


print("Creation COMPLETED !")

np.savetxt("risultati/concorde_len_TSP"+str(config.graph_dimension)+".txt", real_len_concorde)
