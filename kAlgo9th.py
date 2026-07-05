#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy as np
from numpy import pi

from submodules.estimators.efol      import kEfol
from submodules.gcnutils.knavigation import kArrayNav
from submodules.gcnutils.kltisystems import k2OrderLTIsysMimoFactory
from kAlgo89common import kAlgorithm_comboHessian

import kContracts    as ifs
import kLogs         as logs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                        #> algorithm 9th <#                                       #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_bufferSamples(kAlgorithm_comboHessian, ifs.kIF_Algorithm, logs.kLogEfol):

    def __init__(self, 
                 Ts      =  0,
                 q0      =  None,
                 la0     =  None,
                 bufSize =  1,
    ):
        # initial quaternion:
        if q0 is None:
            self.q_s2m = kArrayNav([2,0,2]).to_rad().euler2Q()
        else:
            self.q_s2m = kArrayNav(q0).reshape(4,1)

        # initial lever-arm:
        if la0 is None:
            self.la = kArrayNav([0,0,0], hvector=0)
        else:
            self.la = la0

        # theta vector:
        theta = np.vstack((self.q_s2m, self.la))

        # sampling time:
        self.Ts = Ts

        # size of the buffer:
        self.bufSize = bufSize

        # buffer:
        self.lst_buffer = list()

        # the time will be decimated:
        self.lst_time = list()

        # efol estimator:
        self.efol = kEfol(
                filterpole  = -60,
                dim_error   = 9,
                theta0      = theta,
                Ts          = Ts,
                fn_deadzone = self._deadzone,

                                    #                 q_s2m              la
                Gamma_theta = np.diag( np.hstack(( 3e-4*np.ones(4), 2e-4*np.ones(3) ))),

                                    #                  beta_1          beta_3       beta_2
                Gamma_error = np.diag( np.hstack(( 2e1*np.ones(4), 1e0*np.ones(4),    8e3 ))),
        )

        # preparing to differentiate the measured w_im:
        self.dwim_m = {
                'derivator'  : k2OrderLTIsysMimoFactory(0.6, 2*pi*80, [0,0,0], Ts = Ts,),
                'derivative' : kArrayNav([0,0,0], hvector=0),
        }

        # initialize the logger:
        super().__init__()


    def _update_from_buffer(self, t):
        # calculation of alpha:
        alpha   = kArrayNav(np.vstack((np.zeros((8,1)), [1])))

        # calculating the sums of the elements in the buffer:
        sum_wim_m   = np.zeros((3,1)).view(kArrayNav)
        sum_wim_s   = np.zeros((3,1)).view(kArrayNav)
        sum_fm_m    = np.zeros((3,1)).view(kArrayNav)
        sum_fs_s    = np.zeros((3,1)).view(kArrayNav)
        sum_OM      = np.zeros((3,3)).view(kArrayNav)
        for item in self.lst_buffer:
            sum_wim_m += item['wim_m']
            sum_wim_s += item['wim_s']
            sum_fm_m  += item['fm_m']
            sum_fs_s  += item['fs_s']
            sum_OM    += ( item['wim_m'].to_skew() * item['wim_m'].to_skew() ) + item['dwim_m'].to_skew()

        # calculation of beta:
        beta1 = (sum_wim_s.to_Mplus() - sum_wim_m.to_Mminus()) * self.q_s2m
        beta2 = ((sum_fm_m + (sum_OM*self.la)).to_Mminus() - sum_fs_s.to_Mplus()) * self.q_s2m
        beta3 = self.q_s2m.T * self.q_s2m
        beta  = np.vstack((beta1, beta2, beta3))

        # hessian matrix:
        H = self._hessian(sum_wim_m, sum_wim_s, sum_fm_m, sum_fs_s, sum_OM)

        # updating:
        theta = self.efol.update( alpha, beta, H )
        self.q_s2m = theta[:4]
        self.la    = theta[4:]

        # logging:
        theta = np.vstack((self.q_s2m, self.la))
        self.append(t, alpha, beta, self.efol.get_filtered_error(), theta, self.efol.get_last_ddt())

    def _deadzone_beta1(self, err):
        if err < 0.1:
            maskGain = [0,0,0,0]
        elif err > 2:
            maskGain = ( 1e-1 + np.zeros(4) ).tolist()
        else:
            maskGain = [1,1,1,1]
        return maskGain

    def _deadzone_beta2(self, err):
        if (err < 0.5):
            maskGain = [0,0,0,0]
        elif (err > 10):
            maskGain = ( 1e-2 + np.zeros(4) ).tolist()
        else:
            maskGain = [1,1,1,1]
        return maskGain

    def _deadzone_beta3(self, err):
        if err < 5e-3:
            maskGain = [0]
        else:
            maskGain = [1]
        return maskGain

    def _deadzone(self, err_filt):
        # see self._beta() for the 3 equations.
        maskGain = list()

        norm_err_beta1 = kArrayNav(err_filt[:4]).norm()
        norm_err_beta2 = kArrayNav(err_filt[4:8]).norm()
        norm_err_beta3 = abs(err_filt[8]).reshape(-1)[0]

        maskGain  = self._deadzone_beta1(norm_err_beta1)
        maskGain += self._deadzone_beta2(norm_err_beta2)
        maskGain += self._deadzone_beta3(norm_err_beta3)

        return maskGain


    def update(self, t, wim_m, wim_s, fm_m, fs_s):

        # derivative of wim_m (this shall be updated for each new sample)
        self.dwim_m['derivator'].update(wim_m)
        self.dwim_m['derivative'] = self.dwim_m['derivator'].deinterleave( self.dwim_m['derivator'].get_state() )[3:]
        self.dwim_m['derivative'] = kArrayNav( self.dwim_m['derivative'] )

        # add new inputs to the buffer:
        self.lst_buffer.append({
            "wim_m"  :  wim_m,
            "wim_s"  :  wim_s,
            "fm_m"   :  fm_m,
            "fs_s"   :  fs_s,
            "dwim_m" : self.dwim_m['derivative'],
        })

        # are there enough samples in the buffer to recalculate theta?
        if len(self.lst_buffer) < self.bufSize:
            return self.q_s2m, self.la  # return the same last update

        # perform an update step based on the available buffer:
        self._update_from_buffer(t)

        # cleaning buffer:
        self.lst_buffer.clear()
        #self.lst_buffer.pop(0)

        return self.q_s2m, self.la

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

