#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy  as np
import pandas as pd

from submodules.gcnutils.knavigation import kArrayNav

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kFileLoading:
    def __init__(self, file_gz):

        self.df = pd.read_csv(file_gz)

        # estimated sampling period:
        self.T  = self.df['T'].to_numpy() # time vector for the simulation
        self.Ts = np.diff(self.T).mean()

    def __len__(self):
        return len(self.df)

    def get_sampling_time(self):
        return self.Ts

    def get_time_array(self):
        return self.T

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kImu (kFileLoading):
    def __init__(self, file_gz):
        """
        ['T', 'acc_x', 'acc_y', 'acc_z', 'wib_x', 'wib_y', 'wib_z']
        """
        super().__init__(file_gz)

    def get_acc(self, idx):
        return kArrayNav(self.df.iloc[idx][['acc_x', 'acc_y', 'acc_z']].to_numpy(), hvector=False)

    def get_wib(self, idx):
        return kArrayNav(self.df.iloc[idx][['wib_x', 'wib_y', 'wib_z']].to_numpy(), hvector=False)

    def gen_acc(self):
        for idx in range(len(self.df)):
            yield self.get_acc(idx)

    def gen_wib(self):
        for idx in range(len(self.df)):
            yield self.get_wib(idx)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kBodyGroundTruth (kFileLoading):
    def __init__(self, file_gz):
        """
        ['T', 'vN', 'vE', 'vD', 'vNp', 'vEp', 'vDp', 'lat', 'lon', 'h', 'phi_nb', 'tta_nb', 'psi_nb']
        """
        super().__init__(file_gz)

    def get_vNED(self, idx):
        return kArrayNav(self.df.iloc[idx][['vN', 'vE', 'vD']].to_numpy(), hvector=False)

    def get_vNEDp(self, idx):
        return kArrayNav(self.df.iloc[idx][['vNp', 'vEp', 'vDp']].to_numpy(), hvector=False)

    def get_llh(self, idx):
        return kArrayNav(self.df.iloc[idx][['lat', 'lon', 'h']].to_numpy(), hvector=False)

    def get_euler(self, idx):
        return kArrayNav(self.df.iloc[idx][['phi_nb', 'tta_nb', 'psi_nb']].to_numpy(), hvector=False)

    def get_debug(self, idx):
        return {
                'gravity_n': kArrayNav([
                                    self.df.iloc[idx]['debug_gln_0'],
                                    self.df.iloc[idx]['debug_gln_1'],
                                    self.df.iloc[idx]['debug_gln_2'], ], hvector=False),
                'dLat'      : self.df.iloc[idx]['debug_dLat'],
                'dLon'      : self.df.iloc[idx]['debug_dLon'],
                'acc_n'     : kArrayNav([
                                    self.df.iloc[idx]['debug_accn_0'],
                                    self.df.iloc[idx]['debug_accn_1'],
                                    self.df.iloc[idx]['debug_accn_2'], ], hvector=False),
                'acc_b'     : kArrayNav([
                                    self.df.iloc[idx]['debug_accb_0'],
                                    self.df.iloc[idx]['debug_accb_1'],
                                    self.df.iloc[idx]['debug_accb_2'], ], hvector=False),
        }

    def gen_vNED(self):
        for idx in range(len(self.df)):
            yield self.get_vNED(idx)

    def gen_llh(self):
        for idx in range(len(self.df)):
            yield self.get_llh(idx)

    def gen_euler(self):
        for idx in range(len(self.df)):
            yield self.get_euler(idx)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kImuIntegration (kImu):
    """
    This object performs IMU data integration WITHOUT any correction
    based on sensor modeling or similar.
    """

    def __init__(self,
                 file_gz,
                 euler0_n2b = [0,0,0], # [rad]
                 v0_e_b     = [0,0,0], # [m/s]
                 llh0       = [0,0,0], # [rad;rad;m]
        ):
        """
        euler0_n2b  :   [rad]
                        initial value for the euler angles:
                        (phi,tta,psi)
                        from "navigation" to "body" frames.

        v0_e_b      :   [m/s]
                        initial value for the velocity over ground,
                        resolved at "body" frame

        llh0        :   [rad] initial latitude
                        [rad] initial longitude
                        [m]   initial altitude

        """
        # load the file:
        super().__init__(file_gz)

        # initial conditions:
        self.q_n2b      = kArrayNav(euler0_n2b).euler2Q()
        self.vNED       = self.q_n2b.q_inv().q_x_3d(v0_e_b)
        self.llh        = kArrayNav(llh0)

        # last calculated derivative:
        self.ddt = None

    def _trapezoidal_integration(self, f_k_1, f_k, delta):
        """
        Solves:

            \\int f(t) dt, t_{k-1} < t < t_{k}

        Inputs:
            f_k_1   :    f(t_{k-1})
            f_k     :    f(t_{k})
            delta   :    t_{k} - t_{k-1}

        """

        return (delta/2.0) * (f_k_1 + f_k)


    def update(self, gt=None, idx=None):
        """
        Inputs:
            idx     :   (int) index of this step.
                        If not provided, the function takes the last and
                        increments it.

            gt      :   ground truth, used by the tests when internal variables are replaced
                        by external ones.
        """

        # we need an idx to get acc and wib
        if idx is None:
            # get the index from the last derivative plus one:
            if self.ddt is None:
                idx = 0
            else:
                idx = self.ddt['idx'] + 1

        #self.q_n2b = gt.get_euler(idx).euler2Q()
        #self.vNED  = gt.get_vNED(idx).copy()
        #self.llh   = gt.get_llh(idx).copy()

        # current derivative:
        ddt = self._calc_ddt(idx)
        #ddt['vp_e_n'] = gt.get_vNEDp(idx)
        self.debug = ddt['debug']

        # If this is the first ddt calculated, there is not increment to
        # integrate. We will store the derivative:
        if self.ddt is None:
            self.ddt = ddt

        else:
            ddt_k_1     = self.ddt
            delta       = self.Ts * (idx - ddt_k_1['idx'])

            # the trapezoidal integration does not work properly with the derivative of quaternions.
            #self.q_n2b += self._trapezoidal_integration(ddt_k_1['dq_n2b'], ddt['dq_n2b'], delta)
            self.q_n2b += delta * ddt['dq_n2b'] # <= using euler
            self.q_n2b  = self.q_n2b.q_norm()   # <= this is important!

            self.vNED  += self._trapezoidal_integration(ddt_k_1['vp_e_n'], ddt['vp_e_n'], delta)
            self.llh   += self._trapezoidal_integration(ddt_k_1['dllh'],   ddt['dllh'],   delta)

            # and save the new "last" ddt:
            self.ddt = ddt

        return idx


    def _calc_ddt(self, idx):
        wib_b       = self.get_wib(idx)
        f_b         = self.get_acc(idx)
        lat,lon,h   = self.llh
        vN, vE, vD  = self.vNED
        q_b2n       = self.q_n2b.q_inv()

        dLatdt      = kArrayNav.dLat_dt(vN, lat, h)
        dLondt      = kArrayNav.dLong_dt(vE, lat, h)
        wie_n       = kArrayNav.w_ie_n(lat)
        wen_n       = kArrayNav.w_en_n(dLatdt, dLondt, lat)
        g_local_n   = kArrayNav.gravity_n(lat,h)

        dq_n2b      = self.q_n2b.dqdt(wib_b)
        vp_e_n      = q_b2n.q_x_3d(f_b) + g_local_n - (((2*wie_n)+wen_n).to_skew() * self.vNED)
        dllh        = kArrayNav.dLLH_dt(vN, vE, vD, lat, h)


        # debug:
        d1 = q_b2n.q_x_3d(f_b)
        d2 = g_local_n

        debug = {
                'gravity_n' : d2.squeeze(),
                'dLat'      : dLatdt,
                'dLon'      : dLondt,
                'acc_n'     : d1.squeeze(),
                'acc_b'     : f_b.squeeze(),
        }

        # calculate the derivative vector (in continuous):
        fk = {
            'idx'   : idx,
            'dq_n2b': dq_n2b,
            'vp_e_n': vp_e_n,
            'dllh'  : dllh,
            'debug' : debug,
        }

        return fk

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

