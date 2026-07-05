#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy as np

from submodules.estimators.efol      import kEfol
from submodules.gcnutils.knavigation import kArrayNav

import kContracts    as ifs
import kLogs         as logs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                           #> algorithm 4th <#                                    #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_efol_AplusBminus_q2(ifs.kIF_Algorithm, logs.kLogEfol):
    """

    0   =   (wib^s+  -  wib^m-) . q_s2m
    1   =   q_s2m.T  x  q_s2m

    """

    def __init__(self,
                 Ts   =  0,
                 q0   =  None,
    ):
        # initial state:
        if q0 is None:
            self.q_s2m = kArrayNav([0,0,2]).to_rad().euler2Q()
        else:
            self.q_s2m = q0

        # sampling time:
        self.Ts = Ts

        # alpha = [0(x3); 1.0]
        # beta = [ (alfa+ - beta-).q; ||q||^2 ]
        self.efol = kEfol(
                filterpole  = -40,
                dim_error   = 5,
                theta0      = self.q_s2m,
                Ts          = Ts,
                Gamma_theta = 1e-2 * np.eye(4),
                Gamma_error = np.diag( np.hstack((np.ones(4), 1e3)) )
        )

        # initialize the logger:
        super().__init__()

    def _hessian(self, wib_master, wib_slave, q):
        q0, q1, q2, q3 = q
        wm1, wm2, wm3  = wib_master
        ws1, ws2, ws3  = wib_slave

        H = np.vstack((
            [0, wm1 - ws1, wm2 - ws2, wm3 - ws3],
            [-wm1 + ws1, 0, -wm3 - ws3, wm2 + ws2],
            [-wm2 + ws2, wm3 + ws3, 0, -wm1 - ws1],
            [-wm3 + ws3, -wm2 - ws2, wm1 + ws1, 0],
            [2*q0, 2*q1, 2*q2, 2*q3],
        ))

        return H

    def update(self, t, wib_master, wib_slave):
        # calculation of alpha:
        alpha = kArrayNav( [0,0,0,0,1], hvector=0 )

        # calculation of beta:
        q    = self.q_s2m
        beta = kArrayNav( np.vstack(((wib_slave.to_Mplus() - wib_master.to_Mminus())*q, q.T*q)))

        # hessian matrix:
        H = self._hessian(wib_master, wib_slave, q)

        # updating:
        self.q_s2m = self.efol.update( alpha, beta, H )

        # logging:
        self.append(t, alpha, beta, self.efol.get_filtered_error(), self.q_s2m, self.efol.get_last_ddt())

        return self.q_s2m

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)

        if ret is None:
            if (idx == "q_s2m") or (idx == "quat"):
                return super().__getitem__("theta")
            elif idx == "euler":
                return np.asarray( [i.q_norm().Q2euler().to_deg().to_list() for i in self.lst_theta] )
            elif idx == "norm":
                return np.asarray( [i.T*i for i in self.lst_theta] )
            else:
                raise KeyError()
        else:
            return ret

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

