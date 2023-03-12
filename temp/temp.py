import numpy as np

u_arr = [[1,1], [1, 1]]
K = 5
vals = [1, 2, 3, 4]

# print(len(u_arr))
# print(len(u_arr[0]))

tmp = [K - len(u_arr[i]) - len(vals) for i in range(len(u_arr))]
tmp = np.array(tmp)
tmp = np.where(tmp < 0, K, tmp)
if np.min(tmp) >= K:
    print("New user added")
print(tmp)


def get_user_arrays(data, L, K):
    users = np.unique(data["license_plate"])
    user_arrays = [[]]
    for u in users:
        counter = 0
        user_data = data[data["license_plate"]==u]["speed"].values
        stop_cond = np.minimum(len(user_data), L)
        remaining_spaces = [L - len(user_arrays[i]) - len(user_data) for i in range(len(user_arrays))]
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
    return user_arrays