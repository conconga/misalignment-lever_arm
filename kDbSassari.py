#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#

import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import multiprocessing   as mp

from scipy.io import loadmat
from kImu     import kImu
from kLocalConfig import kLocalConfig

from submodules.gcnutils.knavigation import kArrayNav

#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#


"""

This dataset provides the magneto-inertial signals from six MIMU (2 Xsens, 2
APDM, 2 Shimmer) and orientation from 8 reflective markers (VICON) at 3
different speeds (slow, medium, fast).

Proprietary orientations from MIMU vendors are also included. All data are
synchronized at 100 Hz.

Xsens   - MTx      = XS1, XS2
APDM    - Opal     = AP1, AP2
Shimmer - Shimmer3 = SH1, SH2

For each MIMU dataset (XS1, XS2, AP1, AP2, SH1, SH2):

    columns 1     = time vector (or packet counter vector)
    columns 2:4   = accelerometer data (x,y,z) (m/s^2)
    columns 5:7   = gyroscope data (x,y,z) (rad/s)
    columns 8:10  = magnetometer data (x,y,z) (a.u.)
    columns 11:14 = proprietary orientation

Rotations sequence are in the timeframe contained in indz (first rotation),
indx (second rotation), indy (third rotation), and indarb (3D rotation).

Qs (q0, qx, qy, qz) is the orientation obtained by applying the SVD technique
to eight marker position data [A. Cappozzo, A. Cappello, U. D. Croce, and F.
Pensalfini, “Surface-marker cluster design criteria for 3-d bone movement
reconstruction,” IEEE Trans. Biomed. Eng., vol.  44, no. 12, pp. 1165–1174,
1997]

wVicon is the angular velocity obtained by Qs [Chardonnens, J.; Favre, J.;
Aminian, K. An effortless procedure to align the local frame of an inertial
measurement unit to the local frame of another motion capture system. J.
Biomech. 2012,45, 2297–300.]

"""

class kDbSassari (kImu):
    """
    This object will provide the same interface as an object 'kImu' to access acc and gyros,
    but it will use as the source of data the "Sassari DB".

    The original data will be transformed to have z-axis downwards.
    """

    semaphore_convertion = mp.Semaphore(1)

    def __init__(self, localconfig):
        """
        Arguments:

            localconfig :       (kLocalConfig)
        """

        # instance of local configuration:
        self.cfg = localconfig

        # labels:
        self.labels_acc = ['acc_x', 'acc_y', 'acc_z']
        self.labels_wib = ['wib_x', 'wib_y', 'wib_z']

        # clip the data in this range:
        #   fast    :  60 - 130  [s]
        #   medium  :  60 - 140  [s]
        #   slow    :  60 - 180  [s]
        if localconfig.acqMode == "fast":
            self.tmin = 60
            self.tmax = 130
        elif localconfig.acqMode == "medium":
            self.tmin = 60
            self.tmax = 140
        elif localconfig.acqMode == "slow":
            self.tmin = 60
            self.tmax = 180

    def __getitem__(self, idx):
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
        self.semaphore_convertion.acquire()

        if idx == "master":
            self._convert_or_reload(self.cfg[idx], self.cfg.master)

        elif idx == "slave":
            self._convert_or_reload(self.cfg[idx], self.cfg.slave)

        self.semaphore_convertion.release()
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

        return self

    def _convert_or_reload(self, path, sensor_name):
        sensor_name = sensor_name.upper()

        # if the converted file is available, do nothing:
        if os.path.isfile(path):
            # load the file:
            print(f"--> loading file '{path}' already available ...")
            super().__init__(path)

        else:
            # convert the data and create the file:
            # dataframe for the respective data (file and sensor name)
            print(f"--> converting data to '{path}', sensor '{sensor_name}'...")
            temp_df = pd.DataFrame(loadmat(self.cfg['mat'])[sensor_name],
                                   columns = [ 'time',
                                       'acc_x', 'acc_y', 'acc_z',
                                       'wib_x', 'wib_y', 'wib_z',
                                       'mag_x', 'mag_y', 'mag_z',
                                       'q0', 'q1', 'q2', 'q3',
                                    ])

            # sampling frequency
            self.Fs = 100 # [Hz]
            self.Ts = 1./self.Fs # [s]

            # time vector (the column "time" is a 16-bits counter)
            T = np.arange(len(temp_df))/self.Fs

            # The data is useless at the beginning and at the end, with no movement along
            # several second. We will cut them out.
            # clipping useless data:
            temp_df = temp_df[ (T >= self.tmin) & (T <= self.tmax) ]

            # time vector after clipping:
            self.T  = np.arange(0,len(temp_df)) / self.Fs

            # extract relevant coluns:
            temp_df = temp_df[[ 'acc_x', 'acc_y', 'acc_z', 'wib_x', 'wib_y', 'wib_z' ]]

            # the dataset has Z upwards. We will rotate 180[deg] around X:
            q = kArrayNav([180,0,0]).to_rad().euler2Q()
            self.df = temp_df.apply(lambda row: self._rotate_row(row, q), axis=1)

            # adding the column with time:
            self.df = pd.concat((pd.DataFrame(self.T, columns=["T"]), self.df.reset_index()), axis=1)

            # exporting the generated data:
            print("  > exporting ...")
            self.df.to_csv(path, index=False, compression="gzip")


    def _rotate_row(self, row, q):
        acc = row[self.labels_acc].to_numpy()
        wib = row[self.labels_wib].to_numpy()

        acc_rotated = q.q_x_3d(acc)
        wib_rotated = q.q_x_3d(wib)

        return pd.Series( np.vstack((acc_rotated, wib_rotated)).reshape(-1).tolist(),
                         index=(self.labels_acc+self.labels_wib))

    def plot(self, ax, tag):
        if tag == "acc":
            ax.plot(self.T, self.df[self.labels_acc])
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("accelerometer [m/s2]")

        elif tag == "wib":
            ax.plot(self.T, self.df[self.labels_wib])
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("gyroscope [rad/s]")


def fn_fig_for_a_sensor(sensor, ds, nb_fig):
    Fs = 100
    T  = np.arange(0,ds.shape[0])/Fs

    plt.figure(nb_fig).clf()
    fig, ax = plt.subplots(4,1, num=nb_fig, sharex=True)
    ax = ax.reshape(-1)
    plt.figure(fig).canvas.manager.set_window_title(f"[{sensor}] all data")

    # columns 2:4   = accelerometer data (x,y,z) (m/s^2)
    ax[0].plot(T,ds[:, 1:4])
    ax[0].grid(True, alpha=0.3)
    ax[0].set_ylabel("accelerometer [m/s2]")

    # columns 5:7   = gyroscope data (x,y,z) (rad/s)
    ax[1].plot(T,ds[:, 4:7])
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylabel("gyroscope [rad/s]")

    # columns 8:10  = magnetometer data (x,y,z) (a.u.)
    ax[2].plot(T,ds[:, 7:10])
    ax[2].grid(True, alpha=0.3)
    ax[2].set_ylabel("magnetometer")

    # columns 11:14 = proprietary orientation
    ax[3].plot(T,ds[:, 10:14])
    ax[3].grid(True, alpha=0.3)
    ax[3].set_ylabel("orientation")

#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
def do_main():

    # showing all data from a file for sensor 'XS1':
    data    = loadmat(os.path.join(kLocalConfig.path_sassari_imu, "slow_v4.mat"))
    sensors = [ 'XS1', 'XS2', 'AP1', 'AP2', 'SH1', 'SH2', ]
    for nb_fig, sensor in enumerate(sensors):
        fn_fig_for_a_sensor(sensor, data[sensor], nb_fig=nb_fig)

    # testing the object kDbSassari:
    cfg = kLocalConfig(
            dataSource = "sassari",
            acqMode    = "medium",
            master     = "XS1", 
            slave      = "XS1", # indifferent...
    )
    sassari = kDbSassari(cfg)['master']
    nb_fig += 1

    plt.figure(nb_fig).clf()
    fig, ax = plt.subplots(2,1, num=nb_fig, sharex=True)
    ax = ax.reshape(-1)
    plt.figure(fig).canvas.manager.set_window_title("[XS1] acc+wib")
    sassari.plot(ax[0], "acc")
    sassari.plot(ax[1], "wib")

    #===========================================#
    for fig in plt.get_fignums():
        plt.figure(fig).canvas.flush_events()
        plt.figure(fig).canvas.draw()

    plt.show(block=False)

#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#

if __name__ == "__main__":
    do_main()

#//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
