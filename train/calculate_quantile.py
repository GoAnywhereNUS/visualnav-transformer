import pandas as pd
import numpy as np
from scipy.stats import rankdata
from filterpy.kalman import KalmanFilter
np.set_printoptions(precision=2, suppress=True)

path = '/home/zishuo/GNM_VAE/train/kl_calibtration.csv'
# read csv without header
data = pd.read_csv(path, header=None)
print(data)

def _init_kalman(x):
    my_filter = KalmanFilter(dim_x=1, dim_z=1)
    my_filter.x = x
    my_filter.F = np.eye(1)
    my_filter.H = np.eye(1)
    my_filter.P = 1 * np.eye(1)
    my_filter.R = 100 * np.eye(1)     # observation noise
    my_filter.Q = 0.01 * np.eye(1)    # process noise
    return my_filter

data = data.to_numpy()

##################################
# data = data.flatten()
# data = data
# # rearrange data from high to low 
# sorted_idx = np.argsort(data)
# sorted_arr = data[sorted_idx]
# print(sorted_arr)
# np.save('quantile_rnd.npy', sorted_arr)
# # print(sorted_arr[int(len(sorted_arr) * 0.8)])
##################################

######################################
data_new = np.zeros((data.shape[0], data.shape[1]))
for traj_idx in range(data.shape[1]):
    traj_data = data[:, traj_idx]
    filter = _init_kalman(traj_data[0])
    data_new[0, traj_idx] = filter.x
    for i in range(1, len(traj_data)):
        filter.predict()
        filter.update(traj_data[i])
        x = filter.x
        data_new[i, traj_idx] = x

data = data_new
n1_data = data[:, :int(data.shape[1] / 2)]
mean_n1 = n1_data.mean(axis=1)
n2_data = data[:, int(data.shape[1] / 2):]
n2_data = n2_data - mean_n1[:, np.newaxis]
max_timestep = n2_data.max(axis=0)
sorted_idx = np.argsort(max_timestep)
sorted_arr = max_timestep[sorted_idx]

print(sorted_arr)
h_90 = sorted_arr[int(len(sorted_arr) * 0.9)]
print(h_90)
print(mean_n1 + h_90)
print(mean_n1)

# save sorted_arr as npy
np.save('quantile_gnm_kl.npy', sorted_arr)
np.save('mean_n1_gnm_kl.npy', mean_n1)
a = np.mean(mean_n1 + h_90)
print(a)

########################################