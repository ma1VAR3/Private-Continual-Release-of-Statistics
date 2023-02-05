import math
import numpy as np


def get_left_right_counts(values):
    left_counts = []
    right_counts = []
    
    for v in values:
        left_counts.append(len([x for x in values if x < v]))
        right_counts.append(len([x for x in values if x > v]))
        
    return left_counts, right_counts

def get_probs(values, left_counts, right_counts, epsilon):
    c = np.maximum(left_counts, right_counts)
    probs = [math.exp((-epsilon*c[i])/4) for i in range(len(values))]
    probs = probs / np.sum(probs)
    print("Probability assigned to quantized means: ", probs)
    return probs


def private_median_of_means(means, l, l_b, u_b, epsilon):
    
    # Creating bins for quantization
    quantized_bins = np.arange(l_b, u_b, l)
    quantized_bins = np.append(quantized_bins, u_b) if quantized_bins[-1] != u_b else quantized_bins
    
    # Assiging each mean to the closest bin
    diff_matrix = np.subtract.outer(means, quantized_bins)
    idx = np.abs(diff_matrix).argmin(axis=1)
    quantized_means = quantized_bins[idx]
    
    quantized_means = np.sort(quantized_means)
    left_counts, right_counts = get_left_right_counts(quantized_means)
    print("Quantized bins: ", quantized_bins)
    print("Quantized means: ", quantized_means)
    
    probs = get_probs(quantized_means, left_counts, right_counts, epsilon)
    selected_quantized_mean = np.random.choice(quantized_means, p=probs)
    
    return selected_quantized_mean


if __name__ == "__main__":
    means = [1, 2, 3, 4, 5]
    l = 1.5
    l_b = 0
    u_b = 10
    epsilon = 1
    private_mean = private_median_of_means(means, l, l_b, u_b, epsilon)    
    print("Actual mean of means: ", np.mean(means))
    print("Actual median of means: ", np.median(means))
    print("Private median of means: ", private_mean)
    print("Epsilon: ", epsilon)
