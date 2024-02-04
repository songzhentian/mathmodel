import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

file_path = 'D:\\MCM\\code\\forest\\决赛数据_games.csv'

play1_count = ['TPW_dif', 'COMPLETE_dif','rank_dif','ATPP_dif','AAG_dif', 'SERVEADV','res_dif','speed','serve_width','serve_depth','return_depth']
data1 = pd.read_csv(file_path, usecols=play1_count)
play1 = data1.values
play1[:,6] *= 10000

play2_count = ['TPW_dif', 'COMPLETE_dif','rank_dif','ATPP_dif','AAG_dif', 'SERVEADV','res_dif','speed','serve_width','serve_depth','return_depth']
data2 = pd.read_csv(file_path, usecols=play2_count)
play2 = data2.values
play2[:,6] *= 10000
play2[:,0:7] *= -1

win_count = ['game_victor']
win = pd.read_csv(file_path, usecols=win_count)
win = win.values

#输出play1的数据
win1=[]
for i in win:
    if i==1:
        win1.append(1)
    else:
        win1.append(0)
win1=np.array(win1)
win1_column = win1.reshape(-1, 1)
play1_with_out = np.hstack((play1, win1_column))
column_names1 = play1_count + ['win/lose']
play1_out = pd.DataFrame(play1_with_out, columns=column_names1)
play1_out.to_csv('play1.csv', index=False)


#输出play2的数据
win2=[]
for i in win:
    if i==2:
        win2.append(1)
    else:
        win2.append(0)
win2=np.array(win2)
win2_column = win2.reshape(-1, 1)
play2_with_out = np.hstack((play2, win1_column))
column_names2 = play2_count + ['win/lose']
play2_out = pd.DataFrame(play2_with_out, columns=column_names2)
play2_out.to_csv('play2.csv', index=False)