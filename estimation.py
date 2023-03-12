import math
import os
import numpy as np

def get_left_right_counts(bins, values):
    left_counts = []
    right_counts = []
    
    for v in bins:
        left_counts.append(len([x for x in values if x < v]))
        right_counts.append(len([x for x in values if x > v]))
        
    return left_counts, right_counts

def get_probs(bins, left_counts, right_counts, epsilon, factor):
    c = np.maximum(left_counts, right_counts)
    probs = [math.exp((-epsilon*c[i])/2 * factor) for i in range(len(bins))]
    probs = probs / np.sum(probs)
    # print("Probability assigned to quantized means: ", probs)
    return probs

def get_probs_quantiles(vals, alpha, epsilon):
    k = len(vals)
    probs = [(vals[i+1] - vals[i])*(math.exp(-epsilon*(i - (alpha*k))))for i in range(len(vals)-1)]
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

def private_estimation(user_group_means, K, ub, lb, epsilon, num_exp, actual_mean, groupping_algo, conc_algo, config):
    
    file_base = './results/' + conc_algo + '/' + groupping_algo + '/' + 'epsilon_' + str(epsilon) + '/'
    
    if conc_algo == "coarse_mean":
        taus = config["tau"]
        for tau in taus:
            file_base_tau = file_base + 'tau_' + str(tau) + '/'
            os.makedirs(file_base_tau, exist_ok=True)
            # Quantizing means
            quantized_bins = np.arange(lb+tau/2, ub, tau)
            factor = 2 if groupping_algo == "wrap" else 1
            diff_matrix = np.subtract.outer(user_group_means, quantized_bins)
            idx = np.abs(diff_matrix).argmin(axis=1)
            quantized_means = quantized_bins[idx]
            quantized_means = np.sort(quantized_means)
            
            # Assigning probabilities to quantized means
            left_counts, right_counts = get_left_right_counts(quantized_bins, quantized_means)
            probs = get_probs(quantized_bins, left_counts, right_counts, epsilon/2, factor)
            
            # Selecting quantized means and projecting them
            selected_quantized_means = np.random.choice(quantized_bins, num_exp, p=probs)
            ub_calc = selected_quantized_means + 2*tau
            lb_calc = selected_quantized_means - 2*tau
            lb_calc = [tau/2 if l < tau/2 else (65-(4*tau)) if l > (65-(4*tau)) else l for l in lb_calc]
            ub_calc = [(4.5)*tau if ub < (4.5)*tau else 65 if ub > 65 else ub for ub in ub_calc]
            projected_vals = [np.clip(user_group_means, lb_calc[i], ub_calc[i]) for i in range(len(lb_calc))]
            mean_of_projected_vals = np.mean(projected_vals, axis=1)
            noise_projected_vals = np.random.laplace(0, (4*tau*factor)/(K*(epsilon/2)), num_exp)
            final_estimates = mean_of_projected_vals + noise_projected_vals
            
            # Calculating losses
            losses = np.abs(final_estimates - actual_mean)
            np.save(file_base_tau + 'losses.npy', losses)
            statistical_losses = np.abs(mean_of_projected_vals - actual_mean)
            np.save(file_base_tau + 'statistical_losses.npy', statistical_losses)
            random_losses = np.abs(noise_projected_vals)
            np.save(file_base_tau + 'random_losses.npy', random_losses)
    
    elif conc_algo == "quantiles":
        quantiles = config["lower_quantile"]
        for q in quantiles:
            file_base_q = file_base + 'lq_' + str(q) + '/'
            os.makedirs(file_base_q, exist_ok=True)
            user_group_means = np.append(user_group_means, lb) if lb not in user_group_means else user_group_means
            user_group_means = np.append(user_group_means, ub) if ub not in user_group_means else user_group_means
            user_group_means = np.sort(user_group_means)
            factor = 2 if groupping_algo == "wrap" else 1
            quantile_1 = q
            quantile_2 = 1 - q
            q1_t = private_quantile(user_group_means, quantile_1, epsilon/4, ub, lb, num_exp)
            q2_t = private_quantile(user_group_means, quantile_2, epsilon/4, ub, lb, num_exp)
            q1 = np.minimum(q1_t, q2_t)
            q2 = np.maximum(q1_t, q2_t)
            projected_vals = [np.clip(user_group_means, q1[i], q2[i]) for i in range(len(q1))]
            mean_of_projected_vals = np.mean(projected_vals, axis=1)
            noise_projected_vals = [np.random.laplace(0, ( ((q2[i]-q1[i])*factor) / K * (epsilon/2))) for i in range(len(q1))]
            final_estimates = mean_of_projected_vals + noise_projected_vals
            losses = np.abs(final_estimates - actual_mean)
            np.save(file_base_q + 'losses.npy', losses)
            statistical_losses = np.abs(mean_of_projected_vals - actual_mean)
            np.save(file_base_q + 'statistical_losses.npy', statistical_losses)
            random_losses = np.abs(noise_projected_vals)
            np.save(file_base_q + 'random_losses.npy', random_losses)
    