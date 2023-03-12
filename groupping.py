import numpy as np


def get_user_arrays(data, L, exp_type):
    user_arrays = None
    K = None
    if exp_type == "wrap":
        users = np.unique(data["User"])
        data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
        K = np.floor(np.sum([np.minimum(i, L) for i in data_grouped["Value"]]) / L)
        user_arrays = np.zeros((int(K), int(L)))
        arr_idx = 0
        val_idx = 0
        for u in users:
            counter = 0
            user_data = data[data["User"]==u]["Value"].values
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
        users = np.unique(data["User"])
        user_arrays = [[]]
        for u in users:
            counter = 0
            user_data = data[data["User"]==u]["Value"].values
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