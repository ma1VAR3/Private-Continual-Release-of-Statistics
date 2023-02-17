import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_left_right_counts(bins, values):
    left_counts = []
    right_counts = []
    
    for v in bins:
        left_counts.append(len([x for x in values if x < v]))
        right_counts.append(len([x for x in values if x > v]))
        
    return left_counts, right_counts

def get_probs(bins, left_counts, right_counts, epsilon):
    c = np.maximum(left_counts, right_counts)
    probs = [math.exp((-epsilon*c[i])/4) for i in range(len(bins))]
    probs = probs / np.sum(probs)
    # print("Probability assigned to quantized means: ", probs)
    return probs


def private_median_of_means(means, l, l_b, u_b, epsilon):
    
    # Creating bins for quantization
    bin_size = 2 * math.pow(2, -l/2)
    quantized_bins = np.arange(l_b, u_b, bin_size)
    quantized_bins = np.append(quantized_bins, u_b) if quantized_bins[-1] != u_b else quantized_bins
    
    # Assiging each mean to the closest bin
    diff_matrix = np.subtract.outer(means, quantized_bins)
    idx = np.abs(diff_matrix).argmin(axis=1)
    quantized_means = quantized_bins[idx]
    
    quantized_means = np.sort(quantized_means)
    left_counts, right_counts = get_left_right_counts(quantized_bins, quantized_means)
    # print("Quantized bins: ", quantized_bins)
    # print("Quantized means: ", quantized_means)
    
    probs = get_probs(quantized_bins, left_counts, right_counts, epsilon)
    selected_quantized_mean = np.random.choice(quantized_bins, p=probs)
    
    return selected_quantized_mean


if __name__ == "__main__":
    
    l = 6
    l_b = -10
    u_b = 60
    epsilon = 0.1
    beta = 0.01
    num_exp = 1000
    k = math.floor((16 / epsilon) * (math.log(  math.pow(2, l/2) / beta  )))
    print("Number of user groups (k): ", k) 
    num_vals = int(k * math.pow(2, l-1))
    
    # print("Means of user groups: ", means)
    
    cum_loss = []
    for i in range(num_exp):
        vals = np.random.normal(18.56960, 10.332769, num_vals)
        grouped_vals = np.array_split(vals, k)
        means = [np.mean(x) for x in grouped_vals]
        private_mean = private_median_of_means(means, l, l_b, u_b, epsilon) 
        # print("ACTUAL MEAN: ", np.mean(means))
        # print("PRIVATE MEAN: ", private_mean)
        loss = np.abs(private_mean - np.mean(means))
        print("LOSS: ", loss)
        cum_loss.append(loss)
    
    print("Cumulative loss: ", np.mean(cum_loss))
    
    # fig = go.Figure()
    
    # fig.add_trace(go.Histogram(x=cum_loss, nbinsx=100))
    # fig.add_trace(go.Histogram(x=cum_loss, nbinsx=100, cumulative_enabled=True))
    # fig.update_traces(opacity=0.6)
    # fig.show()
    # print("Actual mean of means: ", np.mean(means))
    # print("Actual median of means: ", np.median(means))
    # print("Cumulative loss for emperical mean: ", cum_loss)
    # print("Private median of means: ", private_mean)
    print("Epsilon: ", epsilon)
