#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

from numpy import pi

from submodules.gcnutils.knavigation import kArrayNav
from submodules.gcnutils.kltisystems import k2OrderLTIsysMimoFactory
from submodules.estimators.rls       import kForgettingFactorRLS

import kContracts    as ifs
import kLogs         as logs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_rls_leverarm(ifs.kIF_Algorithm, logs.kLogRls):
    def __init__(self,
                 Ts   =  0,
                 la0  =  None,
    ):

        # initial state:
        if la0 is None:
            self.la = kArrayNav([0,0,0], hvector=0)
        else:
            self.la = la0

        # sampling time:
        self.Ts = Ts

        # RLS instance:
        self.rls = kForgettingFactorRLS(kArrayNav([0,0,0], hvector=0), lbd=1.000)

        # preparing to differentiate the measured w_im:
        self.dwim_m = {
                'derivator'  : k2OrderLTIsysMimoFactory(0.6, 2*pi*20, [0,0,0], Ts = Ts,),
                'derivative' : kArrayNav([0,0,0], hvector=0),
        }

        # initialize the logger:
        super().__init__()

    def update(self, wim_m, fm_m, fs_s, q_s2m):
        """
        My notes, eq. (9)
        """

        # derivative of wim_m:
        self.dwim_m['derivator'].update(wim_m)
        self.dwim_m['derivative'] = self.dwim_m['derivator'].deinterleave( self.dwim_m['derivator'].get_state() )[3:]
        self.dwim_m['derivative'] = kArrayNav( self.dwim_m['derivative'] )

        # the matrix OMEGA:
        OM = (wim_m.to_skew() * wim_m.to_skew()) + self.dwim_m['derivative'].to_skew()

        # LHR:
        alpha = fm_m - q_s2m.q_x_3d(fs_s)

        # RLS update:
        self.la = self.rls.update(alpha, -OM)

        # RHS:
        beta  = -OM * self.la

        # logging:
        self.append(alpha, beta)

        return self.la

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)

        if ret is None:
            raise KeyError()
        else:
            return ret

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
