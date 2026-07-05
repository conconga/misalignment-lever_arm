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
#                        #> algorithm 8th <#                                       #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_combo(kAlgorithm_comboHessian, ifs.kIF_Algorithm, logs.kLogEfol):
    """

      alpha = [0(x8); 1.0]

      beta1 = ([wib^s]+ - [wib^m]-) . qs2m
      beta2 = ([fm + Om.ra]-  -  [fs]+) . qs2m
      beta3 = qs2m.T x qs2m

      theta = [qs2m;la]

    """

    def __init__(self, 
                 Ts   =  0,
                 q0   =  None,
                 la0  =  None,
    ):
        # initial state:
        if q0 is None:
            self.q_s2m = kArrayNav([2,0,2]).to_rad().euler2Q()
        else:
            self.q_s2m = q0

        if la0 is None:
            self.la = kArrayNav([0,0,0], hvector=0)
        else:
            self.la = la0

        # combo vector:
        combo = np.vstack((self.q_s2m, self.la))

        # sampling time:
        self.Ts = Ts

        # efol estimator:
        self.efol = kEfol(
                filterpole  = -60,
                dim_error   = 9,
                theta0      = combo,
                Ts          = Ts,
                fn_deadzone = self._deadzone,

                                    #                 q_s2m             lever-arm
                Gamma_theta = np.diag( np.hstack(( 5e-4*np.ones(4), 1e-4*np.ones(3) ))),

                                    #                 beta_1          beta_2      beta_3
                Gamma_error = np.diag( np.hstack(( 1e1*np.ones(4), 1e0*np.ones(4), 8e3)) )
        )

        # preparing to differentiate the measured w_im:
        self.dwim_m = {
                'derivator'  : k2OrderLTIsysMimoFactory(0.6, 2*pi*80, [0,0,0], Ts = Ts,),
                'derivative' : kArrayNav([0,0,0], hvector=0),
        }

        # initialize the logger:
        super().__init__()

    def _deadzone_beta1(self, err):
        if err < 0.05:
            maskGain = [0,0,0,0]
        elif err > 1.0:
            maskGain = ( 5e-2 + np.zeros(4) ).tolist()
        else:
            maskGain = [1,1,1,1]
        return maskGain

    def _deadzone_beta2(self, err):
        if (err < 0.5):
            maskGain = [0,0,0,0]
        elif (err > 10):
            maskGain = ( 1e-3 + np.zeros(4) ).tolist()
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


    def _beta(self, wim_m, wim_s, fm_m, fs_s, Om):

        q  = self.q_s2m
        la = self.la

        beta1 = (wim_s.to_Mplus() - wim_m.to_Mminus()) * q
        beta2 = ((fm_m + (Om * la)).to_Mminus() - fs_s.to_Mplus()) * q
        beta3 = q.T * q

        beta  = np.vstack((beta1, beta2, beta3))
        return beta

    def update(self, t, wim_m, wim_s, fm_m, fs_s):

        # calculation of alpha:
        alpha   = kArrayNav(np.vstack((np.zeros((8,1)), [1])))

        # derivative of wim_m:
        self.dwim_m['derivator'].update(wim_m)
        self.dwim_m['derivative'] = self.dwim_m['derivator'].deinterleave( self.dwim_m['derivator'].get_state() )[3:]
        self.dwim_m['derivative'] = kArrayNav( self.dwim_m['derivative'] )

        # the matrix OMEGA:
        OM = (wim_m.to_skew() * wim_m.to_skew()) + self.dwim_m['derivative'].to_skew()

        # calculation of beta and hessian:
        beta  = self._beta(wim_m, wim_s, fm_m, fs_s, OM)

        # hessian matrix:
        H = self._hessian(wim_m, wim_s, fm_m, fs_s, OM)

        # updating:
        combo = self.efol.update( alpha, beta, H )
        self.q_s2m = combo[:4]
        self.la    = combo[4:]

        # logging:
        self.append(t, alpha, beta, self.efol.get_filtered_error(), combo, self.efol.get_last_ddt())

        return self.q_s2m, self.la


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

