import math
import pickle
import numpy as np
import pandas as pd
import h3
import plotly.express as px
from dash import Dash, html, dcc
import plotly.graph_objects as go


def get_data():
    df =pd.read_csv("./suratITMSDPtest/suratITMSDPtest.csv")
    df = df.drop_duplicates(subset=['trip_id', 'observationDateTime'], ignore_index=True)
    df = df.drop(columns = [
                "trip_direction",
                "last_stop_id",
                "last_stop_arrival_time",
                "route_id",
                "actual_trip_start_time",
                "trip_delay",
                "vehicle_label",
                "id",
                "location.type",
                "trip_id"
            ])
    # separating latitude and longitude from location
    lat_lon = df["location.coordinates"].astype(str).str.strip('[]').str.split(",")
    lon = lat_lon.apply(lambda x: x[0])
    lat = lat_lon.apply(lambda x: x[1])
    # assigning h3 index to the latitude and longitude coordinates in separate dataframe 
    dflen = len(df)
    h3index = [None] * dflen
    resolution = 7
    for i in range(dflen):
        h3index[i] = h3.geo_to_h3(lat=float(lat[i]), lng=float(lon[i]), resolution=resolution)
    df["h3index"] = h3index
    # assigning date and time to separate dataframe and creating a timeslot column
    df["Date"] = pd.to_datetime(df["observationDateTime"]).dt.date
    df["Time"] = pd.to_datetime(df["observationDateTime"]).dt.time
    time = df["Time"]
    df["Timeslot"] = time.apply(lambda x: x.hour)
    # assigning HATs from H3index and timeslot
    df["HAT"] = (df["Timeslot"].astype(str) + " " + df["h3index"])
    # Filtering time slots by start and end time 
    startTime = 9
    endTime = 20
    df = df[(df["Timeslot"] >= startTime) & (df["Timeslot"] <= endTime)]
    # print(df.head(10))
    df = df[df["speed"]>0]
    return df

def get_top_k_diversity_hats(data, k):
    df_gb_h_al = data.groupby(["HAT"]).agg({"license_plate": "nunique"}).reset_index()
    max_hat = df_gb_h_al[df_gb_h_al["license_plate"] == df_gb_h_al["license_plate"].max() ]["HAT"].iloc[0]
    return [max_hat]

def get_hats_data(data, hats):
    hats_data = []
    print("\n"+"="*30+"X"*5+"="*30+"\n")
    for h in hats:
        print("Params for HAT: ", h)
        h_d = data[data["HAT"]==h]
        h_d_gb = h_d.groupby(["HAT", "license_plate"]).agg({"speed":"count"}).reset_index()
        p_sets = []
        m_star = h_d_gb["speed"].max()
        m_avg = math.floor(np.mean(h_d_gb["speed"]))
        m_rms = math.floor(math.sqrt(np.mean([math.pow(i, 2) for i in h_d_gb["speed"]])))
        m_median = np.median(h_d_gb["speed"])
        
        print("Sigma mi : ", np.sum(h_d_gb["speed"]))
        k_mstar = np.floor(np.sum([np.minimum(i, m_star) for i in h_d_gb["speed"]]) / m_star)
        k_mavg = np.floor(np.sum([np.minimum(i, m_avg) for i in h_d_gb["speed"]]) / m_avg)
        k_mrms = np.floor(np.sum([np.minimum(i, m_rms) for i in h_d_gb["speed"]]) / m_rms)
        k_mmedian = np.floor(np.sum([np.minimum(i, m_median) for i in h_d_gb["speed"]]) / m_median)
        
        print("When L=m* => L = {}, K = {}, Samples Used = {}".format(m_star, k_mstar, m_star*k_mstar))
        print("When L=m_rms => L = {}, K = {}, Samples Used = {}".format(m_rms, k_mrms, m_rms*k_mrms))
        print("When L=m_avg => L = {}, K = {}, Samples Used = {}".format(m_avg, k_mavg, m_avg*k_mavg))
        print("When L=m_median => L = {}, K = {}, Samples Used = {}".format(m_median, k_mmedian, m_median*k_mmedian))
        # p_sets.append({
        #     "Type" : "Maximum m",
        #     "L" : m_star,
        #     "K" : k_mstar
        # })
        # p_sets.append({
        #     "Type" : "RMS m",
        #     "L" : m_rms,
        #     "K" : k_mrms
        # })
        p_sets.append({
            "Type" : "Average m",
            "L" : m_avg,
            "K" : k_mavg
        })
        p_sets.append({
            "Type" : "Median m",
            "L" : m_median,
            "K" : k_mmedian
        })
        hat_data = {
            "HAT" : h,
            "data" : h_d,
            "param_sets" : p_sets
        }
        hats_data.append(hat_data)
    return hats_data

def get_user_arrays(data, L, K, exp_type):
    user_arrays = None
    if exp_type == "wrap":
        users = np.unique(data["license_plate"])
        user_arrays = np.zeros((int(K), int(L)))
        arr_idx = 0
        val_idx = 0
        for u in users:
            counter = 0
            user_data = data[data["license_plate"]==u]["speed"].values
            stop_cond = np.minimum(len(user_data), L)
            while counter < stop_cond:
                
                user_arrays[arr_idx][val_idx] = user_data[counter]
                counter += 1
                val_idx += 1
                if val_idx >= L:
                    val_idx = 0
                    arr_idx += 1
                if arr_idx >= K:
                    break
            if arr_idx >= K:
                break
    
    elif exp_type == "best_fit":
        users = np.unique(data["license_plate"])
        user_arrays = [[]]
        for u in users:
            counter = 0
            user_data = data[data["license_plate"]==u]["speed"].values
            stop_cond = np.minimum(len(user_data), L)
            remaining_spaces = [L - len(user_arrays[i]) - stop_cond for i in range(len(user_arrays))]
            remaining_spaces = np.array(remaining_spaces)
            remaining_spaces = np.where(remaining_spaces < 0, L, remaining_spaces)
            array_idx_to_fill = None
            if np.min(remaining_spaces) >= L:
                user_arrays.append([])
                array_idx_to_fill = -1
            else:
                array_idx_to_fill = np.argmin(remaining_spaces)    
            while counter < stop_cond:
                user_arrays[array_idx_to_fill].append(user_data[counter])
                counter += 1
                
        K = len(user_arrays)
    
    return user_arrays, K
    

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
    selected_interval = np.random.choice(indices, num_exp, p=probs)
    selected_quantile = [np.random.uniform(new_s_vals[selected_interval[i]], new_s_vals[selected_interval[i]+1]) for i in range(len(selected_interval))]
    return selected_quantile

def private_estimation(user_group_means, L, K, tau, ub, lb, epsilon, num_exp, actual_mean, exp_type1, exp_type2):
    losses = None
    statistical_losses = None
    random_losses = None
    if exp_type2 == "mean":
        quantized_bins = np.arange(lb+tau/2, ub, tau)
        factor = 2 if exp_type1 == "wrap" else 1
        diff_matrix = np.subtract.outer(user_group_means, quantized_bins)
        idx = np.abs(diff_matrix).argmin(axis=1)
        quantized_means = quantized_bins[idx]
        quantized_means = np.sort(quantized_means)
        left_counts, right_counts = get_left_right_counts(quantized_bins, quantized_means)
        probs = get_probs(quantized_bins, left_counts, right_counts, epsilon/2, factor)
        selected_quantized_means = np.random.choice(quantized_bins, num_exp, p=probs)
        ub_calc = selected_quantized_means + 2*tau
        lb_calc = selected_quantized_means - 2*tau
        lb_calc = [tau/2 if l < tau/2 else (65-(4*tau)) if l > (65-(4*tau)) else l for l in lb_calc]
        ub_calc = [(4.5)*tau if ub < (4.5)*tau else 65 if ub > 65 else ub for ub in ub_calc]
        projected_vals = [np.clip(user_group_means, lb_calc[i], ub_calc[i]) for i in range(len(lb_calc))]
        mean_of_projected_vals = np.mean(projected_vals, axis=1)
        noise_projected_vals = np.random.laplace(0, (4*tau*factor)/(K*(epsilon/2)), num_exp)
        final_estimates = mean_of_projected_vals + noise_projected_vals
        losses = np.abs(final_estimates - actual_mean)
        statistical_losses = np.abs(mean_of_projected_vals - actual_mean)
        random_losses = np.abs(noise_projected_vals)
    
    elif exp_type2 == "quant":
        user_group_means = np.append(user_group_means, lb) if lb not in user_group_means else user_group_means
        user_group_means = np.append(user_group_means, ub) if ub not in user_group_means else user_group_means
        user_group_means = np.sort(user_group_means)
        factor = 2 if exp_type1 == "wrap" else 1
        quantile_1 = 0.05
        quantile_2 = 0.95
        q1_t = private_quantile(user_group_means, quantile_1, epsilon/4, ub, lb, num_exp)
        q2_t = private_quantile(user_group_means, quantile_2, epsilon/4, ub, lb, num_exp)
        q1 = np.minimum(q1_t, q2_t)
        q2 = np.maximum(q1_t, q2_t)
        projected_vals = [np.clip(user_group_means, q1[i], q2[i]) for i in range(len(q1))]
        mean_of_projected_vals = np.mean(projected_vals, axis=1)
        noise_projected_vals = [np.random.laplace(0, ( ((q2[i]-q1[i])*factor) / K * (epsilon/2))) for i in range(len(q1))]
        final_estimates = mean_of_projected_vals + noise_projected_vals
        losses = np.abs(final_estimates - actual_mean)
        statistical_losses = np.abs(mean_of_projected_vals - actual_mean)
        random_losses = np.abs(noise_projected_vals)
    
    return losses, statistical_losses, random_losses

    
def get_table_row(L_v, K_v, tau, percentile, val, val_stat, val_random, type, s1, s2):
    if type == "main":
        return html.Tr([
            html.Td(L_v, rowSpan=s1),
            html.Td(K_v, rowSpan=s1),
            html.Td(tau, rowSpan=s2),
            html.Td(round(val_stat, 3), rowSpan=s2),
            html.Td(round(val_random, 3), rowSpan=s2),
            html.Td(percentile, className="tbl-row1"),
            html.Td(round(val, 3), className="tbl-row2")
        ])
    
    elif type == "sub":
        return html.Tr([
            html.Td(tau, rowSpan=s2),
            html.Td(round(val_stat, 3), rowSpan=s2),
            html.Td(round(val_random, 3), rowSpan=s2),
            html.Td(percentile, className="tbl-row1"),
            html.Td(round(val, 3), className="tbl-row2")
        ])
        
    return html.Tr([
        html.Td(percentile, className="tbl-row1"),
        html.Td(round(val, 3), className="tbl-row2")
    ])
    
def get_table(L_v, K_v, tau, percentiles, vals, vals_stat, vals_rand):
    t_children = []
    t_children.append(
        html.Thead([
            html.Tr([
                html.Th("L"),
                html.Th("K"),
                html.Th("Tau"),
                html.Th("Statistical Error"),
                html.Th("Random Error"),
                html.Th("Percentile"),
                html.Th("MAE Loss Val")
            ])
        ])
    )
    counter = 0
    type="main"
    tb_children = []
    for i in range(len(L_v)):
        for j in range(len(tau)):
            for k in range(len(percentiles)):
                if counter % (len(tau) * len(percentiles)) == 0:
                    type = "main"
                elif counter % len(percentiles) == 0:
                    type = "sub"
                else:
                    type = "subsub"
                tb_children.append(get_table_row(L_v[i], K_v[i], tau[j], percentiles[k], vals[i][j][k], vals_stat[i][j], vals_rand[i][j], type, len(tau) * len(percentiles), len(percentiles)))
                counter += 1
    t_children.append(html.Tbody(tb_children))
    return html.Table(t_children)

if __name__ == "__main__":
    experiment_type1 = "best_fit"
    experiment_type2 = "mean"
    epsilons = np.arange(0.1, 1.1, 0.1)
    beta = 0.01
    tau = [2, 3, 4, 5]
    upper_bound = 65
    lower_bound = 0
    num_hats = 1
    num_exp = 100000
    percentiles = [95, 98, 99]
    figs = []
    tables = []
    lap_bounds = []
    data = get_data()
    hats = get_top_k_diversity_hats(data, num_hats)
    hats_data = get_hats_data(data, hats)
    print("\n\n"+"="*30+"X"*5+"="*30+"\n")
    for i in range(num_hats):
        hat_dict = hats_data[i]
        print("Running experiments for HAT: ", hat_dict["HAT"])
        hat_data = hat_dict["data"]
        actual_mean = np.mean(hat_data["speed"].values)
        print("Actual mean: ", actual_mean)
        print()
        for epsilon in epsilons:
            plot_x = []
            plot_y = []
            plot_zt = []
            mech_bounds = []
            mech_bounds_stat = []
            mech_bounds_random = []
            print("Running experiments for epsilon: ", epsilon)
            for params in hat_dict["param_sets"]:
                print("\tRunning experiment for parameter settings: ")
                print("\tType: {}, L: {}, K: {}".format(params["Type"], params["L"], params["K"]))
                user_arrays, calc_K = get_user_arrays(hat_data, params["L"], params["K"], experiment_type1)
                params["K"] = calc_K
                plot_x.append(params["L"])
                plot_y.append(params["K"])
                actual_means_of_user_groups = [np.mean(x) for x in user_arrays]
                print("\tMean of user groups: ", np.mean(actual_means_of_user_groups))
                plot_data = []
                param_err_bounds = []
                param_err_bounds_stat = []
                param_err_bounds_random = []
                for t in tau:
                    print("\tTau=", t)
                    err_bounds = []
                    losses, statistical_losses, random_losses = private_estimation(
                        actual_means_of_user_groups,
                        params["L"],
                        params["K"],
                        t,
                        upper_bound,
                        lower_bound,
                        epsilon,
                        num_exp,
                        actual_mean,
                        experiment_type1,
                        experiment_type2
                    )
                    print("\t\tAverage MAE across all runs: ", np.mean(losses))
                    print("\t\tAverage statistical loss across all runs: ", np.mean(statistical_losses))
                    print("\t\tAverage random loss across all runs: ", np.mean(random_losses))
                    for p in percentiles:
                        print("\t\t{}th Percentile MAE across all runs: ".format(p), np.percentile(losses, p))
                        err_bounds.append(np.percentile(losses, p))
                    plot_data.append(np.mean(losses))
                    param_err_bounds.append(err_bounds)
                    param_err_bounds_stat.append(np.mean(statistical_losses))
                    param_err_bounds_random.append(np.mean(random_losses))
                plot_zt.append(plot_data)
                mech_bounds.append(param_err_bounds)
                mech_bounds_stat.append(param_err_bounds_stat)
                mech_bounds_random.append(param_err_bounds_random)
                print()    
                
            #Generate table here:
            tab = get_table(plot_x, plot_y, tau, percentiles, mech_bounds, mech_bounds_stat, mech_bounds_random)
            tables.append(tab)
            
            plot_z = []
            for k in range(len(plot_zt[0])):
                tau_vals = []
                for l in range(len(plot_zt)):
                    tau_vals.append(plot_zt[l][k])
                plot_z.append(tau_vals)
            
            fig = go.Figure()
            for k in range(0, len(plot_z)):
                fig.add_trace(
                    go.Scatter3d(x=plot_x, y=plot_y, z=plot_z[k], name="Tau="+str(tau[k]))
                )
            fig.update_layout(
                scene = dict(
                    xaxis_title='L',
                    yaxis_title='K',
                    zaxis_title='MAE'
                )
            )
            
            fig2 = px.imshow(plot_zt, x=tau, y=plot_x, labels=dict(x="Tau", y="L"), text_auto=True, aspect="auto")
            
            f_arr = [fig, fig2]
            figs.append(f_arr)
            
            # Running experiments for Laplace mechanism
            print("\tRunning experiments for Laplace mechanism")
            h_d_gb = hat_data.groupby(["HAT", "license_plate"]).agg({"speed":"count"}).reset_index()
            m_star = h_d_gb["speed"].max()
            sigma_mi = np.sum(h_d_gb["speed"])
            lap_losses = []
            for j in range(num_exp):
                noise = np.random.laplace(0, (upper_bound - lower_bound) * m_star/(sigma_mi*epsilon))
                lap_losses.append(np.abs(noise))
            print("\t\tAverage MAE across all runs: ", np.mean(lap_losses))
            lap_bounds_p = []
            for p in percentiles:
                print("\t\t{}th Percentile MAE across all runs: ".format(p), np.percentile(lap_losses, p))
                lap_bounds_p.append(np.percentile(lap_losses, p))
            lap_bounds.append(lap_bounds_p)
            print()
    
    f_prefix = "./saved/" + experiment_type2 + "/" + experiment_type1 + "/"
            
    with open(f_prefix + "tables.pkl", 'wb') as f:
        pickle.dump(tables, f)
    
    with open(f_prefix + "figs.pkl", 'wb') as f:
        pickle.dump(figs, f)
        
    with open(f_prefix + "lap_bounds.pkl", 'wb') as f:
        pickle.dump(lap_bounds, f)
    
    with open(f_prefix + "epsilons.pkl", 'wb') as f:
        pickle.dump(epsilons, f)
        
    with open(f_prefix + "percentiles.pkl", 'wb') as f:
        pickle.dump(percentiles, f)
        