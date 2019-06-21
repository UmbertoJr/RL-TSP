import pickle
from dataGenerator import DataGenerator
from utils import visualize_2D_trip

last = 19000
i = 19100
seed_ = 19053
with open("supervised_data/batch_seed_"+str(last)+"_"+str(i)+".pkl", 'rb') as handle:
    b = pickle.load(handle)


dataset = DataGenerator()
input_batch, dist_batch = dataset.create_batch(256, 50, 2, seed=seed_)

visualize_2D_trip(input_batch[100], b[seed_][100], 'niente', 1)
