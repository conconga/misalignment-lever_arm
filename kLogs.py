#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy         as np
import kContracts    as ifs
from submodules.gcnutils.knavigation import kArrayNav

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kLogEfol(ifs.kIF_Log):

    def __init__(self):
        self.lst_alpha      = list()
        self.lst_beta       = list()
        self.lst_err_filt   = list()
        self.lst_err_unfilt = list()
        self.lst_theta      = list()
        self.lst_dtheta     = list()
        self.lst_time       = list()

    def append(self, t, alpha, beta, err_filt, theta, dtheta):
        self.lst_alpha.append(alpha.copy())
        self.lst_beta.append(beta.copy())
        self.lst_err_unfilt.append((beta-alpha).copy())
        self.lst_err_filt.append(err_filt.copy())
        self.lst_theta.append(theta.copy())
        self.lst_dtheta.append(dtheta.copy())
        self.lst_time.append(t)

    def __getitem__(self, idx):
        if idx == 'alpha':
            return np.asarray( [i.to_list() for i in self.lst_alpha] )
        elif idx == 'beta':
            return np.asarray( [i.to_list() for i in self.lst_beta] )
        elif idx == 'err_filt':
            return np.asarray( [kArrayNav(i).to_list() for i in self.lst_err_filt] )
        elif idx == 'err_unfilt':
            return np.asarray( [i.to_list() for i in self.lst_err_unfilt] )
        elif idx == 'theta':
            return np.asarray( [i.to_list() for i in self.lst_theta] )
        elif idx == 'dtheta':
            return np.asarray( [i.to_list() for i in self.lst_dtheta] )
        elif idx == "T":
            return np.asarray( self.lst_time )
        else:
            return None

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kLogRls(ifs.kIF_Log):

    def __init__(self):
        self.lst_alpha      = list()
        self.lst_beta       = list()
        self.lst_la         = list()

    def append(self, alpha, beta):
        self.lst_alpha.append(alpha.copy())
        self.lst_beta.append(beta.copy())
        self.lst_la.append(self.la.copy())

    def __getitem__(self, idx):
        if idx == 'alpha':
            return np.asarray( [i.to_list() for i in self.lst_alpha] )
        elif idx == 'beta':
            return np.asarray( [i.to_list() for i in self.lst_beta] )
        elif idx == "la":
            return np.asarray( [kArrayNav(i).to_list() for i in self.lst_la] )
        elif isinstance(idx, int):
            return np.asarray( [i[idx] for i in self.lst_la] )
        else:
            return None

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kLogEig(ifs.kIF_Log):

    def __init__(self):
        self.lst_q_s2m = list()
        self.lst_time  = list()

    def append(self, t):
        self.lst_q_s2m.append(self.q_s2m)
        self.lst_time.append(t)

    def __getitem__(self, idx):
        if (idx == "q_s2m") or (idx == "quat"):
            return np.asarray( [i.to_list() for i in self.lst_q_s2m] )
        elif idx == "euler":
            return np.asarray( [i.q_norm().Q2euler().to_deg().to_list() for i in self.lst_q_s2m] )
        elif idx == "norm":
            return np.asarray( [i.T * i for i in self.lst_q_s2m] )
        elif idx == "T":
            return np.asarray( self.lst_time )
        else:
            return None

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

