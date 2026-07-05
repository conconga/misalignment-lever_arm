#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import numpy as np

from submodules.gcnutils.knavigation import kArrayNav

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                        #> hessian for the combo-algorithms <#                    #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kAlgorithm_comboHessian:
    def _hessian(self, wim_m, wim_s, fm_m, fs_s, OM):
        M11, M12, M13 = OM[0]
        M21, M22, M23 = OM[1]
        M31, M32, M33 = OM[2]
        q0, q1, q2, q3 = self.q_s2m.reshape(-1)
        ra1, ra2, ra3  = self.la.reshape(-1)
        fm1, fm2, fm3  = fm_m.reshape(-1)
        fs1, fs2, fs3  = fs_s.reshape(-1)
        wm1, wm2, wm3  = wim_m.reshape(-1)
        ws1, ws2, ws3  = wim_s.reshape(-1)

        H = np.vstack((
            [0, wm1 - ws1, wm2 - ws2, wm3 - ws3, 0, 0, 0],
            [-wm1 + ws1, 0, -wm3 - ws3, wm2 + ws2, 0, 0, 0],
            [-wm2 + ws2, wm3 + ws3, 0, -wm1 - ws1, 0, 0, 0],
            [-wm3 + ws3, -wm2 - ws2, wm1 + ws1, 0, 0, 0, 0],
            [0, -M11*ra1 - M12*ra2 - M13*ra3 - fm1 + fs1, -M21*ra1 - M22*ra2 - M23*ra3 - fm2 + fs2, -M31*ra1 - M32*ra2 - M33*ra3 - fm3 + fs3, -M11*q1 - M21*q2 - M31*q3, -M12*q1 - M22*q2 - M32*q3, -M13*q1 - M23*q2 - M33*q3],
            [M11*ra1 + M12*ra2 + M13*ra3 + fm1 - fs1, 0, M31*ra1 + M32*ra2 + M33*ra3 + fm3 + fs3, -M21*ra1 - M22*ra2 - M23*ra3 - fm2 - fs2, M11*q0 - M21*q3 + M31*q2, M12*q0 - M22*q3 + M32*q2, M13*q0 - M23*q3 + M33*q2],
            [M21*ra1 + M22*ra2 + M23*ra3 + fm2 - fs2, -M31*ra1 - M32*ra2 - M33*ra3 - fm3 - fs3, 0, M11*ra1 + M12*ra2 + M13*ra3 + fm1 + fs1, M11*q3 + M21*q0 - M31*q1, M12*q3 + M22*q0 - M32*q1, M13*q3 + M23*q0 - M33*q1],
            [M31*ra1 + M32*ra2 + M33*ra3 + fm3 - fs3, M21*ra1 + M22*ra2 + M23*ra3 + fm2 + fs2, -M11*ra1 - M12*ra2 - M13*ra3 - fm1 - fs1, 0, -M11*q2 + M21*q1 + M31*q0, -M12*q2 + M22*q1 + M32*q0, -M13*q2 + M23*q1 + M33*q0],
            [2*q0, 2*q1, 2*q2, 2*q3, 0, 0, 0],
        ))

        return H

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)

        if ret is None:
            if (idx == "q_s2m") or (idx == "quat"):
                return np.asarray( [i.reshape(-1)[:4] for i in self.lst_theta] )
            if (idx == "la"):
                return np.asarray( [i.reshape(-1)[4:] for i in self.lst_theta] )
            elif idx == "euler":
                return np.asarray( [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in self["q_s2m"]] )
            elif idx == "norm":
                return np.asarray( [kArrayNav(i,hvector=1)*kArrayNav(i,hvector=0) for i in self["q_s2m"]] )
            elif idx == "mask_filt":
                err_filt = self['err_filt']
                norm_err_beta1 = [kArrayNav(i[:4]).norm() for i in err_filt]
                norm_err_beta2 = [kArrayNav(i[4:8]).norm() for i in err_filt]
                norm_err_beta3 = [abs(i[8]).reshape(-1)[0] for i in err_filt]
                ret = [ list(i) for i in zip(norm_err_beta1, norm_err_beta2, norm_err_beta3)]
                return np.asarray(ret)
            elif idx == "mask_unfilt":
                err_filt = self['err_unfilt']
                norm_err_beta1 = [kArrayNav(i[:4]).norm() for i in err_filt]
                norm_err_beta2 = [kArrayNav(i[4:8]).norm() for i in err_filt]
                norm_err_beta3 = [abs(i[8]).reshape(-1)[0] for i in err_filt]
                ret = [ list(i) for i in zip(norm_err_beta1, norm_err_beta2, norm_err_beta3)]
                return np.asarray(ret)
            elif idx == "norm-dq":
                return np.asarray( [kArrayNav(i[:4]).norm() for i in self.lst_dtheta] )
            elif idx == "norm-dla":
                return np.asarray( [kArrayNav(i[4:]).norm() for i in self.lst_dtheta] )
            else:
                raise KeyError()
        else:
            return ret

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

