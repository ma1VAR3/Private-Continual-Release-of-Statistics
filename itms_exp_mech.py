import math
import numpy as np
import pandas as pd
import h3

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
        print("Sigma mi : ", np.sum(h_d_gb["speed"]))
        k_mstar = np.floor(np.sum([np.minimum(i, m_star) for i in h_d_gb["speed"]]) / m_star)
        k_mavg = np.floor(np.sum([np.minimum(i, m_avg) for i in h_d_gb["speed"]]) / m_avg)
        k_mrms = np.floor(np.sum([np.minimum(i, m_rms) for i in h_d_gb["speed"]]) / m_rms)
        print("When L=m* => L = {}, K = {}, Samples Used = {}".format(m_star, k_mstar, m_star*k_mstar))
        print("When L=m_rms => L = {}, K = {}, Samples Used = {}".format(m_rms, k_mrms, m_rms*k_mrms))
        print("When L=m_avg => L = {}, K = {}, Samples Used = {}".format(m_avg, k_mavg, m_avg*k_mavg))
        p_sets.append({
            "Type" : "Maximum m",
            "L" : m_star,
            "K" : k_mstar
        })
        p_sets.append({
            "Type" : "RMS m",
            "L" : m_rms,
            "K" : k_mrms
        })
        p_sets.append({
            "Type" : "Average m",
            "L" : m_avg,
            "K" : k_mavg
        })
        hat_data = {
            "HAT" : h,
            "data" : h_d,
            "param_sets" : p_sets
        }
        hats_data.append(hat_data)
    return hats_data

def get_user_arrays(data, L, K):
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
    return user_arrays

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

def private_median_of_means(user_group_means, L, tau, ub, lb, epsilon):
    quantized_bins = [lb]
    quantized_bins = np.append(quantized_bins, np.arange(lb+tau/2, ub, tau))
    quantized_bins = np.append(quantized_bins, ub) if quantized_bins[-1] != ub else quantized_bins
    
    diff_matrix = np.subtract.outer(user_group_means, quantized_bins)
    idx = np.abs(diff_matrix).argmin(axis=1)
    quantized_means = quantized_bins[idx]
    
    quantized_means = np.sort(quantized_means)
    left_counts, right_counts = get_left_right_counts(quantized_bins, quantized_means)
    probs = get_probs(quantized_bins, left_counts, right_counts, epsilon)
    selected_quantized_mean = np.random.choice(quantized_bins, p=probs)
    return selected_quantized_mean

def project_vals(vals, coarse_mean, tau):
    ub = coarse_mean + 2*tau
    lb = coarse_mean - 2*tau
    projected_vals = np.clip(vals, )
    

if __name__ == "__main__":
    
    epsilon = 1
    beta = 0.01
    tau = [4]
    upper_bound = 65
    lower_bound = 0
    num_hats = 1
    num_exp = 1
    data = get_data()
    hats = get_top_k_diversity_hats(data, num_hats)
    hats_data = get_hats_data(data, hats)
    print("\n\n"+"="*30+"X"*5+"="*30+"\n")
    for i in range(num_hats):
        hat_dict = hats_data[i]
        print("Running experiments for HAT: ", hat_dict["HAT"])
        hat_data = hat_dict["data"]
        actual_means = np.mean(hat_data["speed"].values)
        print("Actual mean: ", actual_means)
        print()
        for params in hat_dict["param_sets"]:
            print("Running experiment for parameter settings: ")
            print("\tType: {}, L: {}, K: {}".format(params["Type"], params["L"], params["K"]))
            user_arrays = get_user_arrays(hat_data, params["L"], params["K"])
            actual_means_of_user_groups = [np.mean(x) for x in user_arrays]
            print("\tMean of user groups: ", np.mean(actual_means_of_user_groups))
            for t in tau:
                print("\tTau=", t)
                for j in range(num_exp):
                    mean_coarse_estimate = private_median_of_means(
                        actual_means_of_user_groups, 
                        params["L"], 
                        t, 
                        upper_bound, 
                        lower_bound, 
                        epsilon
                    )
                    print("\t", mean_coarse_estimate)
            