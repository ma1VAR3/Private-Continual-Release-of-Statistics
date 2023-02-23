import math
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
        # m_medianX = m_median * 0.8
        print("Sigma mi : ", np.sum(h_d_gb["speed"]))
        k_mstar = np.floor(np.sum([np.minimum(i, m_star) for i in h_d_gb["speed"]]) / m_star)
        k_mavg = np.floor(np.sum([np.minimum(i, m_avg) for i in h_d_gb["speed"]]) / m_avg)
        k_mrms = np.floor(np.sum([np.minimum(i, m_rms) for i in h_d_gb["speed"]]) / m_rms)
        k_mmedian = np.floor(np.sum([np.minimum(i, m_median) for i in h_d_gb["speed"]]) / m_median)
        # k_medianX = np.floor(np.sum([np.minimum(i, m_medianX) for i in h_d_gb["speed"]]) / m_medianX)
        print("When L=m* => L = {}, K = {}, Samples Used = {}".format(m_star, k_mstar, m_star*k_mstar))
        print("When L=m_rms => L = {}, K = {}, Samples Used = {}".format(m_rms, k_mrms, m_rms*k_mrms))
        print("When L=m_avg => L = {}, K = {}, Samples Used = {}".format(m_avg, k_mavg, m_avg*k_mavg))
        print("When L=m_median => L = {}, K = {}, Samples Used = {}".format(m_median, k_mmedian, m_median*k_mmedian))
        # print("When L=m_medianX => L = {}, K = {}, Samples Used = {}".format(m_medianX, k_medianX, m_medianX*k_medianX))
        # p_sets.append({
        #     "Type" : "Maximum m",
        #     "L" : m_star,
        #     "K" : k_mstar
        # })
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
        p_sets.append({
            "Type" : "Median m",
            "L" : m_median,
            "K" : k_mmedian
        })
        # p_sets.append({
        #     "Type" : "Median m*0.8",
        #     "L" : m_medianX,
        #     "K" : k_medianX
        # })
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
    quantized_bins = np.arange(lb+tau/2, ub, tau)
    # quantized_bins = np.append(quantized_bins, np.arange(lb+tau/2, ub, tau))
    # quantized_bins = np.append(quantized_bins, ub) if quantized_bins[-1] != ub else quantized_bins
    
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
    if lb<tau/2:
        lb = tau/2
        ub = lb + 4*tau
    if ub>65:
        ub = 65
        lb = 65 - 4*tau
    projected_vals = np.clip(vals, lb, ub)
    return projected_vals
    
def get_table_row(L_v, K_v, tau, percentile, val, type, s1, s2):
    if type == "main":
        return html.Tr([
            html.Td(L_v, rowSpan=s1),
            html.Td(K_v, rowSpan=s1),
            html.Td(tau, rowSpan=s2),
            html.Td(percentile),
            html.Td(val)
        ])
    
    elif type == "sub":
        return html.Tr([
            html.Td(tau, rowSpan=s2),
            html.Td(percentile),
            html.Td(val)
        ])
        
    return html.Tr([
        html.Td(percentile),
        html.Td(val)
    ])
    
def get_table(L_v, K_v, tau, percentiles, vals):
    t_children = []
    t_children.append(
        html.Thead([
            html.Tr([
                html.Th("L"),
                html.Th("K"),
                html.Th("Tau"),
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
                tb_children.append(get_table_row(L_v[i], K_v[i], tau[j], percentiles[k], vals[i][j][k], type, len(tau) * len(percentiles), len(percentiles)))
                counter += 1
    t_children.append(html.Tbody(tb_children))
    return html.Table(t_children)

if __name__ == "__main__":
    
    epsilons = [0.1, 0.5, 1, 2, 5]
    beta = 0.01
    tau = [3, 4, 5]
    upper_bound = 65
    lower_bound = 0
    num_hats = 1
    num_exp = 100000
    percentiles = [95, 98, 99]
    app = Dash(__name__)
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
            print("Running experiments for epsilon: ", epsilon)
            for params in hat_dict["param_sets"]:
                print("\tRunning experiment for parameter settings: ")
                print("\tType: {}, L: {}, K: {}".format(params["Type"], params["L"], params["K"]))
                plot_x.append(params["L"])
                plot_y.append(params["K"])
                user_arrays = get_user_arrays(hat_data, params["L"], params["K"])
                actual_means_of_user_groups = [np.mean(x) for x in user_arrays]
                print("\tMean of user groups: ", np.mean(actual_means_of_user_groups))
                plot_data = []
                param_err_bounds = []
                for t in tau:
                    print("\tTau=", t)
                    losses = []
                    statistical_losses = []
                    random_losses = []
                    err_bounds = []
                    for j in range(num_exp):
                        mean_coarse_estimate = private_median_of_means(
                            actual_means_of_user_groups, 
                            params["L"], 
                            t, 
                            upper_bound, 
                            lower_bound, 
                            epsilon
                        )
                        projected_vals = project_vals(actual_means_of_user_groups, mean_coarse_estimate, t)
                        mean_projected_vals = np.mean(projected_vals)
                        # print("\t\tMean of projected values: ", mean_projected_vals)
                        noise_projected_vals = np.random.laplace(0, (8*t)/(params["K"]*epsilon))
                        final_estimate = mean_projected_vals + noise_projected_vals
                        losses.append(np.abs(final_estimate - actual_mean))
                        statistical_losses.append(np.abs(mean_projected_vals - actual_mean))
                        random_losses.append(np.abs(noise_projected_vals))
                    print("\t\tAverage MAE across all runs: ", np.mean(losses))
                    print("\t\tAverage statistical loss across all runs: ", np.mean(statistical_losses))
                    print("\t\tAverage random loss across all runs: ", np.mean(random_losses))
                    for p in percentiles:
                        print("\t\t{}th Percentile MAE across all runs: ".format(p), np.percentile(losses, p))
                        err_bounds.append(np.percentile(losses, p))
                    plot_data.append(np.mean(losses))
                    param_err_bounds.append(err_bounds)
                plot_zt.append(plot_data)
                mech_bounds.append(param_err_bounds)
                print()    
                
            #Generate table here:
            tab = get_table(plot_x, plot_y, tau, percentiles, mech_bounds)
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
                noise = np.random.laplace(0, (upper_bound - lower_bound) * m_star/(sigma_mi*2*epsilon))
                lap_losses.append(np.abs(noise))
            print("\t\tAverage MAE across all runs: ", np.mean(lap_losses))
            lap_bounds_p = []
            for p in percentiles:
                print("\t\t{}th Percentile MAE across all runs: ".format(p), np.percentile(lap_losses, p))
                lap_bounds_p.append(np.percentile(lap_losses, p))
            lap_bounds.append(lap_bounds_p)
            print()
            
    
    children = [
        html.H1(children='One Shot Exponential Mechanism for Mean Estimation'),

        html.Div(children='Based on research conucted in [FS17], [Lev+21] and [GRST22]', className="description"),
    ]
    
    for f in range (len(figs)):
        lap_children = []
        for p in range(len(percentiles)):
            lap_children.append(
                html.P(children='For Laplace Mechanism, {}th Percentile MAE: '.format(percentiles[p])+str(lap_bounds[f][p]), className="laplace-bounds")
            )
        children.append(
            html.Div([
                html.P(children='For Epsilon: '+str(epsilons[f]), className="epsilon"),
                html.Div([
                        dcc.Graph(
                        id='3D-plot'+str(f),
                        figure=figs[f][0],
                        className="line-plot"
                    ),
                    dcc.Graph(
                        id='heatmap-plot'+str(f),
                        figure=figs[f][1],
                        className="heatmap-plot"
                    )
                ], className="graph-container"),
                html.Div([tables[f]], className="table-container"),
                html.Div(children=lap_children, className="lap-bounds-container")
            ], className="epsilon-container")
        )
        
    app.layout = html.Div(children=children)
        
    app.run_server(debug=True)