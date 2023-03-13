import math
import os
import numpy as np

import plotly.express as px

def get_probs_quantiles(vals, alpha, epsilon):
    k = len(vals) - 1
    probs = [  (vals[i+1] - vals[i])  *  (  math.exp(  -epsilon  *  abs(i - (alpha*k))   )  )  for i in range(len(vals)-1)]
    probs = probs / np.sum(probs)
    
    return probs

def private_quantile(vals, q, epsilon, ub, lb, num_vals):
    vals_c = [lb if v < lb else ub if v > ub else v for v in vals]
    vals_sorted = np.sort(vals_c)
    new_s_vals = [lb]
    new_s_vals = np.append(new_s_vals, vals_sorted)
    new_s_vals = np.append(new_s_vals, ub)
    probs = get_probs_quantiles(new_s_vals, q, epsilon)
    indices = np.arange(0, len(new_s_vals)-1)
    selected_interval = np.random.choice(indices, num_vals, p=probs)
    selected_quantile = [np.random.uniform(new_s_vals[selected_interval[i]], new_s_vals[selected_interval[i]+1]) for i in range(len(selected_interval))]
    return selected_quantile

def private_estimation(user_group_means, K, ub, lb, epsilon, num_exp, actual_mean):
    
    q = 0.90
    
        
    print("95th percentile: ", np.percentile(user_group_means, q*100))
    user_group_means = np.append(user_group_means, lb) if lb not in user_group_means else user_group_means
    user_group_means = np.append(user_group_means, ub) if ub not in user_group_means else user_group_means
    user_group_means = np.sort(user_group_means)
    factor = 1
    quantile_1 = q
    quantile_2 = 1 - q
    q1_t = private_quantile(user_group_means, quantile_1, epsilon/4, ub, lb, num_exp)
    fig = px.histogram(q1_t, nbins=30)
    fig.show()
    q2_t = private_quantile(user_group_means, quantile_2, epsilon/4, ub, lb, num_exp)
    q1 = np.minimum(q1_t, q2_t)
    q2 = np.maximum(q1_t, q2_t)
    projected_vals = [np.clip(user_group_means, q1[i], q2[i]) for i in range(len(q1))]
    mean_of_projected_vals = np.mean(projected_vals, axis=1)
    noise_projected_vals = [np.random.laplace(0, ( ((q2[i]-q1[i])*factor) / K * (epsilon/2))) for i in range(len(q1))]
    final_estimates = mean_of_projected_vals + noise_projected_vals
    losses = np.abs(final_estimates - actual_mean)
        
    
if __name__ == "__main__":
    user_group_means = np.load("./user_group_mean.npy")
    epsilons = [0.1, 1, 5, 10]
    K = 87
    actual_mean = 18.46674
    for e in epsilons:
        private_estimation(
            user_group_means,
            K,
            65,
            0,
            e,
            100000,
            actual_mean,
        )