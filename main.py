import json
from utils import load_data, calc_user_array_length
from groupping import get_user_arrays
from estimation import private_estimation, baseline_estimation

import numpy as np

if __name__ == "__main__":
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile) 
        print("Configurations loaded from config.json")
        jsonfile.close()
    dataset = config["dataset"]
    data_non_iid = load_data(dataset, config["data"][dataset])
    # data_iid = 
    data = data_non_iid
    
    L = calc_user_array_length(data, type=config["user_group_size"])
    print("L: ", L)
    user_arrays, K = get_user_arrays(data, L, config["user_groupping"])
    print("K:", K)
    actual_mean = np.mean(data["Value"].values)
    user_group_means = [np.mean(x) for x in user_arrays]
    np.save("./user_group_mean.npy", user_group_means)
    epsilons = config["epsilons"]
    upper_bound = config["data"][dataset]["upper_bound"]
    lower_bound = config["data"][dataset]["lower_bound"]
    num_experiments = config["num_experiments"]
    print("Actual mean: ", actual_mean)
    baseline_estimation(data, upper_bound, lower_bound, epsilons, num_experiments)
    for e in epsilons:
        private_estimation(
            user_group_means,
            L,
            K,
            upper_bound,
            lower_bound,
            e,
            num_experiments,
            actual_mean,
            config["user_groupping"],
            config["concentration_algorithm"],
            config["algorithm_parameters"][config["concentration_algorithm"]]
        )