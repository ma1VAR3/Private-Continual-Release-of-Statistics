import pickle
import os
import json
import numpy as np
from dash import Dash, html, dcc
import plotly.graph_objects as go


def get_figure(x_axis_data, y_axis_data, legend, legend_prefix="",multiple=True):
    fig = go.Figure()
    for i in range(len(y_axis_data)):
        fig.add_trace(
            go.Scatter(x=x_axis_data, y=y_axis_data[i], mode='lines+markers', name=legend_prefix+str(legend[i]))
        )
    return fig

def generate_and_save_figures():
    config = None
    with open("./config.json", "r") as jsonfile:
        config = json.load(jsonfile) 
        print("Configurations loaded from config.json")
        jsonfile.close()
    
    l_f = "losses.npy"
    rl_f = "random_losses.npy"
    sl_f = "statistical_losses.npy"
    f_base_cm = "./results/coarse_mean/{}/epsilon_{}/tau_{}/"
    f_base_q = "./results/quantiles/{}/epsilon_{}/lq_{}/"
    
    groupping_algos = ["wrap", "best_fit"]
    conc_algos = ["coarse_mean", "quantiles"]
    epsilons = config["epsilons"]
    taus = config["algorithm_parameters"]["coarse_mean"]["tau"]
    quants = config["algorithm_parameters"]["quantiles"]["lower_quantile"]
    
    # Coarse Mean plots
    for algo in groupping_algos:
        x_data = epsilons
        y_data_mae = []
        y_data_percentiles = []
        for t in taus:
            tau_results_mae = []
            tau_results_percentiles = []
            for e in epsilons:
                f = f_base_cm.format(algo, e, t)
                losses = np.load(f + l_f)
                tau_results_mae.append(np.mean(losses))
                tau_results_percentiles.append(np.percentile(losses, 95))
            y_data_mae.append(tau_results_mae)
            y_data_percentiles.append(tau_results_percentiles)
        fig_f = './figures/coarse_mean/{}/{}.pkl'
        os.makedirs('./figures/coarse_mean/{}/'.format(algo), exist_ok=True)
        with open(fig_f.format(algo, "mae"), 'wb') as f:
            pickle.dump(get_figure(x_data, y_data_mae, taus, "tau="), f)
        with open(fig_f.format(algo, "percentiles"), 'wb') as f:
            pickle.dump(get_figure(x_data, y_data_percentiles, taus, "tau="), f)
        
        
    # Quantiles plots
    for algo in groupping_algos:
        x_data = epsilons
        y_data_mae = []
        y_data_percentiles = []
        for q in quants:
            q_results_mae = []
            q_results_percentiles = []
            for e in epsilons:
                f = f_base_q.format(algo, e, q)
                losses = np.load(f + l_f)
                q_results_mae.append(np.mean(losses))
                q_results_percentiles.append(np.percentile(losses, 95))
            y_data_mae.append(q_results_mae)
            y_data_percentiles.append(q_results_percentiles)
        fig_f = './figures/quantiles/{}/{}.pkl'
        os.makedirs('./figures/quantiles/{}/'.format(algo), exist_ok=True)
        with open(fig_f.format(algo, "mae"), 'wb') as f:
            pickle.dump(get_figure(x_data, y_data_mae, quants, "lq="), f)
        with open(fig_f.format(algo, "percentiles"), 'wb') as f:
            pickle.dump(get_figure(x_data, y_data_percentiles, quants, "lq="), f)