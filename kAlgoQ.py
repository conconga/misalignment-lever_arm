#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

from submodules.gcnutils.knavigation import kArrayNav

import numpy         as np
import kContracts    as ifs
import kLogs         as logs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_QMethod(ifs.kIF_Algorithm, logs.kLogEig):
    def __init__(self,
                 Ts   =  0,
    ):

        # sampling time:
        self.Ts = Ts

        # "the" matrix, not initialized:
        self.K = None

        # initialize the logger:
        super().__init__()

    def update(self, t, wim_m, wim_s):

        # updating matrix K:
        x = wim_m.to_Mplus() - wim_s.to_Mminus()
        if self.K is None:
            self.K = self.Ts * (x.T * x)
        elif False:
            p = 1.0 - 1e-4
            self.K = (p*self.K) + ((1.0-p)*(x.T * x))
        elif False:
            p = 2000
            M = x.T * x
            K = self.K + M/p
            self.K = K/np.trace(K)
        else:
            self.K += self.Ts * (x.T * x)

        # solving: K.q = Lbd.q
        w, v  = np.linalg.eig(self.K)
        i     = np.argmin(w.real)
        q_m2s = kArrayNav(v[:, i])

        self.q_s2m = q_m2s.q_conj()

        # logging:
        self.append(t)

        return self.q_s2m

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)

        if ret is None:
            raise KeyError()
        else:
            return ret

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
