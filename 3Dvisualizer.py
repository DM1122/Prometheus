import datetime as dt
from libs import NSRDBlib, processlib
import math
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd


drop_zeros = True

def configure_data():
    data_raw = NSRDBlib.get_data('dhi')     # get data_raw
    data_raw.reset_index(drop=True, inplace=True)       # remove year index
    data = data_raw[len(data_raw) - 8760:]      # slice data_raw to most recent year
    data.reset_index(drop=True, inplace=True)       # reset index to begin at 0
    
    print('Data:')
    print(data.head())
    print(data.tail())
    print(data.shape)
    
    return data

def map_coords(data):
    print('Generating 3D dataframe...')
    df_hours = pd.DataFrame(columns=['hour'])
    df_days = pd.DataFrame(columns=['day'])
    for i in range(len(data)):
        df_hours.loc[i] = i%24
        df_days.loc[i] = math.floor(i/24)

    data = pd.concat([df_hours, df_days, data], axis=1, sort=False)

    if (drop_zeros):
        data = data[(data != 0).all(1)]     # filter rows w/ zero(s)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

    print('Data_coords:')
    print(data.head())
    print(data.tail())
    print(data.shape)

    return data


def plot_3D(data):
    print('Plotting 3D visualization...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Day')
    ax.set_zlabel('DHI [W/$m^2$]')
    ax.set_xlim3d(0, 24)
    ax.set_ylim3d(0, 365)
    # ax.view_init(30, -60)

    surf = ax.scatter(data['hour'], data['day'], data['dhi'], c=data['dhi'], cmap='inferno', linewidth=0.1, depthshade=True)
    fig.colorbar(surf)
    
    fig.tight_layout()


    save_plot(fig)
    plt.show()


def save_plot(fig):
    # Save plot to disk
    file_name = os.path.basename(__file__)
    plot_date = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    plot_dir = './plots/'+file_name+'/'+plot_date+'/'
    os.makedirs(plot_dir)
    fig.savefig(plot_dir+'DHI_plot.png')


if __name__ == '__main__':

    data = configure_data()

    data_coords = map_coords(data)

    plot_3D(data_coords)