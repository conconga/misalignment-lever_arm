#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import sys
print( "**************************************" )
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")


import numpy as np
from   numpy import inf
from unittest.mock import patch

from kImu import kImu, kBodyGroundTruth, kImuIntegration
from submodules.gcnutils.knavigation import kArrayNav

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

def fn_set_trace():
    import pudb
    pudb.set_trace()

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class TestClass_kImu:

    def test_load_an_IMU_datafile(self):
        slave  = kImu("tests/imu_leverarm.csv.gz")
        master = kImu("tests/imu_carrier.csv.gz")

    def test_load_ground_truth(self):
        gt = kBodyGroundTruth("tests/gt_carrier.csv.gz")

    def test_get_len(self):
        slave  = kImu("tests/imu_leverarm.csv.gz")
        assert 0.009 < slave.get_sampling_time() < 0.011

    def test_time_array(self):
        slave = kImu("tests/imu_leverarm.csv.gz")

    def test_get_measurements(self):
        slave  = kImu("tests/imu_leverarm.csv.gz")
        assert isinstance(slave.get_acc(0), kArrayNav)
        assert isinstance(slave.get_wib(1), kArrayNav)

    def test_generators(self):
        slave = kImu("tests/imu_leverarm.csv.gz")
        count = 0
        for acc, wib in zip(slave.gen_acc(), slave.gen_wib()):
            count += 1
            assert isinstance(acc, kArrayNav)
            assert isinstance(wib, kArrayNav)

        assert count == len(slave)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class TestClass_kBodyGroundTruth:

    def test_generators(self):
        gt    = kBodyGroundTruth("tests/gt_carrier.csv.gz")
        count = 0
        for vNED, llh, euler in zip( gt.gen_vNED(), gt.gen_llh(), gt.gen_euler() ):
            count += 1
            assert isinstance(vNED,  kArrayNav)
            assert isinstance(llh,   kArrayNav)
            assert isinstance(euler, kArrayNav)

        assert count == len(gt)



#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class TestClass_kImuIntegration:

    def test_calc_ddt(self):
        euler0_n2b = [0,0,0]
        v0_e_b     = [0,0,0]
        llh0       = [0,0,0]

        master = kImuIntegration("tests/imu_carrier.csv.gz",
                                 euler0_n2b = euler0_n2b,
                                 v0_e_b     = v0_e_b,
                                 llh0       = llh0,
        )

        for i in range(3):
            with patch.object(master, "get_acc") as mock_get_acc:
                mock_get_acc.return_value = kArrayNav([13,13,13,])
                ddt = master._calc_ddt(i)

                # test f_n_N == 13
                assert abs(ddt['vp_e_n'][0] - 13) < 1e-10

                # test f_n_D ~ g+13
                assert abs(ddt['vp_e_n'][2] - (9.8+13)) < 0.2

    def test_calc_ddt_dq_n2b(self):
        euler0_n2b = [0,0,0]
        v0_e_b     = [0,0,0]
        llh0       = [0,0,0]

        master = kImuIntegration("tests/imu_carrier.csv.gz",
                                 euler0_n2b = euler0_n2b,
                                 v0_e_b     = v0_e_b,
                                 llh0       = llh0,
        )

        for i in range(3):
            # test first derivatives without any fake measurements:
            ddt = master._calc_ddt(i)
            euler_no_fake = (kArrayNav(euler0_n2b).euler2Q() + (master.get_sampling_time() * ddt['dq_n2b'])).q_norm().Q2euler().to_deg()
            #print("euler_no_fake = ", euler_no_fake)
            for j in range(3):
                assert abs(euler_no_fake.squeeze()[j]) < 1e-3

            # now with additional pushes:
            with patch.object(master, "get_wib") as mock_get_wib:
                mock_get_wib.return_value = kArrayNav([[1,0,0], [0,1,0], [0,0,1]][i])
                ddt = master._calc_ddt(i)
                euler_fake = (kArrayNav(euler0_n2b).euler2Q() + ddt['dq_n2b']).q_norm().Q2euler().to_deg()
                #print("euler_fake = ", euler_fake)
                assert euler_fake.squeeze()[i] > 50.0 # ~60 = 1rad

    def test_calc_ddt_vp_e_n(self):
        euler0_n2b = [0,0,0]
        v0_e_b     = [0,0,0]
        llh0       = [0,0,0]

        master = kImuIntegration("tests/imu_carrier.csv.gz",
                                 euler0_n2b = euler0_n2b,
                                 v0_e_b     = v0_e_b,
                                 llh0       = llh0,
        )

        for i in range(3):
            # test first derivatives without any fake measurements:
            ddt = master._calc_ddt(i)
            #print("ddt = ", ddt)
            vNED_no_fake = kArrayNav(v0_e_b, hvector=False) + (1.0 * ddt['vp_e_n'])
            for j in range(3):
                assert abs(vNED_no_fake.squeeze()[j]) < 0.1

            # now with additional pushes:
            with patch.object(master, "get_acc") as mock_get_acc:
                mock_get_acc.return_value = kArrayNav([[1,0,0], [0,1,0], [0,0,1]][i])
                ddt = master._calc_ddt(i)
                vNED_fake = kArrayNav(v0_e_b, hvector=False) + (1.0 * ddt['vp_e_n'])
                #print("vNED_fake = ", vNED_fake)
                for j in range(3):
                    assert vNED_fake.squeeze()[j] > [
                            [0.9, -inf, 9.6],
                            [-inf, 0.9, 9.6],
                            [-inf, -inf, 9.6+1.0]] [i][j]

    def test_calc_ddt_dllh(self):
        euler0_n2b = [0,0,0]
        v0_e_b     = [0,0,0]
        llh0       = [0,0,0]

        master = kImuIntegration("tests/imu_carrier.csv.gz",
                                 euler0_n2b = euler0_n2b,
                                 v0_e_b     = v0_e_b,
                                 llh0       = llh0,
        )

        for i in range(3):
            # test first derivatives without any fake measurements:
            ddt = master._calc_ddt(i)
            #print("ddt = ", ddt)
            llh_no_fake = kArrayNav(llh0, hvector=False) + (master.get_sampling_time() * ddt['dllh'])
            for j in range(3):
                assert abs(llh_no_fake.squeeze()[j]) < 1e-5

            # now with additional pushes:
            master.vNED = kArrayNav([[1,0,0], [0,1,0], [0,0,1]][i], hvector=False)
            ddt = master._calc_ddt(i)
            llh_fake = kArrayNav(llh0, hvector=False) + (master.get_sampling_time() * ddt['dllh'])
            #print("llh_fake = ", llh_fake)

            eps = 1e-10
            if i == 0:
                assert llh_fake.to_list()[0] > eps
                assert llh_fake.to_list()[1] < eps
                assert llh_fake.to_list()[2] < eps
            elif i == 1:
                assert llh_fake.to_list()[0] < eps
                assert llh_fake.to_list()[1] > eps
                assert llh_fake.to_list()[2] < eps
            elif i == 2:
                assert llh_fake.to_list()[0] < eps
                assert llh_fake.to_list()[1] < eps
                assert llh_fake.to_list()[2] < 0.1


    def test_update(self):
        # Not really a test, but a visualization script.
        # You can update the test files running a02_simu_MasterSlave_toFile.py with
        # new parameters.

        # block the figures?
        block = False

        # load ground truth:
        gt = kBodyGroundTruth("tests/gt_carrier.csv.gz")

        # load test file (generated by a02_simu_MasterSlave_toFile)
        euler0_n2b = gt.get_euler(0)
        v0_e_b     = euler0_n2b.euler2C() * gt.get_vNED(0)
        llh0       = gt.get_llh(0)
        master     = kImuIntegration("tests/imu_carrier.csv.gz", 
                                     euler0_n2b = euler0_n2b,
                                     v0_e_b     = v0_e_b,
                                     llh0       = llh0,
        )

        logs = list()
        for idx in range(len(master)):
            assert master.update(gt) == idx

            logs.append( (
                master.q_n2b.q_norm().Q2euler().squeeze(),       # 0
                master.vNED.squeeze().copy(),                    # 1
                master.llh.squeeze().copy(),                     # 2

                gt.get_euler(idx).squeeze(),                     # 3
                gt.get_vNED(idx).squeeze(),                      # 4
                gt.get_llh(idx).squeeze(),                       # 5

                master.ddt['vp_e_n'].squeeze().copy(),           # 6
                gt.get_vNEDp(idx).squeeze().copy(),              # 7

                master.debug.copy(),                             # 8
                gt.get_debug(idx),                               # 9
            ))

        if 1 == 1:
            import matplotlib.pyplot as plt
            plt.figure(1).clf()
            fig, ax = plt.subplots(3,3,num=1,sharex=True)
            ax = ax.reshape(-1)
            T  = master.get_time_array()[:len(logs)]
            for i in range(3):
                for k in range(3):
                    ax[(3*i)+k].plot(T, [ [ j[i][k], j[i+3][k] ] for j in logs])
                    ax[(3*i)+k].grid(True, alpha=0.5)
                    ax[(3*i)+k].set_ylabel( ['euler', 'vNED', 'llh'][i] )

            plt.figure(2).clf()
            fig, ax = plt.subplots(3,1,num=2,sharex=True)
            for i in range(3):
                ax[i].plot(T, [[j[6][i], j[7][i]] for j in logs])
                ax[i].grid(True)
                ax[i].set_ylabel('vNEDp')

            # < debug > #
            plt.figure(3).clf()
            fig, ax = plt.subplots(4,3,num=3,sharex=True)
            ax = ax.reshape(-1)
            data_master = [i[8]['gravity_n'].to_list() + [float(i[8]['dLat']), float(i[8]['dLon'])] + i[8]['acc_n'].to_list() + i[8]['acc_b'].to_list() for i in logs]
            data_master = np.asarray(data_master)
            data_gt =     [i[9]['gravity_n'].to_list() + [float(i[9]['dLat']), float(i[9]['dLon'])] + i[9]['acc_n'].to_list() + i[9]['acc_b'].to_list() for i in logs]
            data_gt = np.asarray(data_gt)
            for i in range(11):
                ax[i].plot(T, data_master[:,i], T, data_gt[:,i])
                ax[i].grid(True)
                ax[i].set_ylabel([ 'gl_0', 'gl_1', 'gl_2', 'dLat', 'dLon', 'accn_0', 'accn_1', 'accn_2', 'accb_0', 'accb_1', 'accb_2', ][i])

            #--#
            for i in plt.get_fignums():
                plt.figure(i).canvas.flush_events()
                plt.figure(i).canvas.draw()
            plt.show(block=block)


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
