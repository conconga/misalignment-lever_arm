#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy as np

from submodules.estimators.efol      import kEfol
from submodules.gcnutils.knavigation import kArrayNav

import kContracts    as ifs
import kLogs         as logs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                           #> algorithm 2nd <#                                    #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kAlgorithm_efol_Cwib_q2(ifs.kIF_Algorithm, logs.kLogEfol):
    def __init__(self,
                 Ts   =  0,
                 q0   =  None,
    ):

        # initial state:
        if q0 is None:
            self.q_s2m = kArrayNav([0,1,0]).to_rad().euler2Q()
        else:
            self.q_s2m = q0

        # sampling time:
        self.Ts = Ts

        ## beta(t) = [C(q).wib; ||q||^2]
        self.efol = kEfol(
                filterpole  = -40,
                dim_error   = 4,
                theta0      = self.q_s2m,
                Ts          = Ts,
                Gamma_theta = 1e-2 * np.eye(4),
                Gamma_error = np.diag( np.hstack((np.ones(3), 1e3)) )
        )

        # initialize the logger:
        super().__init__()

    def _hessian(self, q, beta):
        q0, q1, q2, q3 = q
        v0, v1, v2, v3 = beta
        H = [
                [2*q0*v0 - 2*q2*v2 + 2*q3*v1,
                 2*q1*v0 + 2*q2*v1 + 2*q3*v2,
                 -2*q0*v2 + 2*q1*v1 - 2*q2*v0,
                 2*q0*v1 + 2*q1*v2 - 2*q3*v0],
                [2*q0*v1 + 2*q1*v2 - 2*q3*v0,
                 2*q0*v2 - 2*q1*v1 + 2*q2*v0,
                 2*q1*v0 + 2*q2*v1 + 2*q3*v2,
                 -2*q0*v0 + 2*q2*v2 - 2*q3*v1],
                [2*q0*v2 - 2*q1*v1 + 2*q2*v0,
                 -2*q0*v1 - 2*q1*v2 + 2*q3*v0,
                 2*q0*v0 - 2*q2*v2 + 2*q3*v1,
                 2*q1*v0 + 2*q2*v1 + 2*q3*v2],
                [ 2*q0, 2*q1, 2*q2, 2*q3, ],
        ]
        return H

    def update(self, t, wib_master, wib_slave):

        # calculation of alpha:
        alpha = kArrayNav( np.vstack((wib_master, 1)), hvector=False )

        # calculation of beta:
        q     = self.q_s2m
        beta  = kArrayNav( np.vstack(( q.q_x_3d(wib_slave), q.T * q )), hvector=False )

        # hessian matrix:
        H = self._hessian(q, beta)

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
