#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import os
import numpy  as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm        import tqdm
from math        import sqrt, ceil

from kImu        import kImu
from kDbSassari  import kDbSassari
from kPdfUtils   import kPdfUtils

from submodules.gcnutils.knavigation import kArrayNav

import kLocalConfig       as klc

from kAlgoQ         import kAlgorithm_QMethod
from kAlgo2nd       import kAlgorithm_efol_Cwib_q2
from kAlgo4th       import kAlgorithm_efol_AplusBminus_q2
from kAlgo8th       import kAlgorithm_combo
from kAlgo9th       import kAlgorithm_bufferSamples
from kAlgoRLS       import kAlgorithm_rls_leverarm

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

def fn_parsing():

    ##########################
    ## parser configuration ##
    ##########################

    parser = argparse.ArgumentParser(
            description="Development of a misalignment correction algorithm.",
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
            '-b', '--block',
            required=False,
            action="store_true",
            dest='is_block',
            default=False,
            help='Used with `plt.show(block=???)`',
    )

    parser.add_argument(
            '-c', '--nb-changes-misalignment',
            required=False,
            metavar="<INT>",
            nargs=1,
            dest="nb_changes_misalignment",
            default=[6],
            help   = "how many time the misalignment will randomly change",
    )

    parser.add_argument(
            '-d', '--data-source',
            required=True,
            metavar = "<simulated | sassari>",
            nargs   = 1,
            choices = ['simulated', 'sassari'],
            dest    = "dataSource",
            default = None,
            help    = "selection of the data-source type",
    )

    parser.add_argument(
            '-m', '--acquisition-mode',
            required=False,
            metavar = "<slow | medium | fast>",
            nargs   = 1,
            choices = ['slow', 'medium', 'fast'],
            dest    = "acqMode",
            default = [None],
            help    = "selection of the acquisition mode",
    )

    parser.add_argument(
            '-M', '--master',
            required=False,
            metavar = "< xs1 | xs2 | ap1 | ap2 | sh1 | sh2 >",
            nargs   = 1,
            choices = [ 'xs1', 'xs2', 'ap1', 'ap2', 'sh1', 'sh2', ],
            dest    = "master",
            default = [None],
            help    = "selection of the master sensor",
    )

    parser.add_argument(
            '-S', '--slave',
            required=False,
            metavar = "< xs1 | xs2 | ap1 | ap2 | sh1 | sh2 >",
            nargs   = 1,
            choices = [ 'xs1', 'xs2', 'ap1', 'ap2', 'sh1', 'sh2', ],
            dest    = "slave",
            default = [None],
            help    = "selection of the slave sensor",
    )

    parser.add_argument(
            '--tmax',
            required = False,
            metavar  = "<SECONDS>",
            nargs    = 1,
            dest     = "tmax",
            default  = [None],
            help     = "max duration of the simulation",
    )

    parser.add_argument(
            '--pdf',
            required = False,
            metavar  = "<filename.pdf>",
            nargs    = 1,
            dest     = "pdffile",
            default  = [None],
            help     = "export all open figures to a single pdf file",
    )


    parser.add_argument(
            '-v', '--version',
           action='version',
           version='v. 0.1',
    )

    ####################
    ## parsing inputs ##
    ####################
    args = parser.parse_args()

    ## fixing pdf filename without extension ##
    if args.pdffile[0] is not None:
        args.pdffile[0] = kPdfUtils.fix_filename_extension(args.pdffile[0])

    print(". Parsing arguments:")
    print("args = ", args)
    print()


    if args.dataSource[0] == 'sassari':
        missing = [x for x in ['acqMode', 'master', 'slave'] if getattr(args, x) is None]
        if missing:
            parser.error('With "-d/--data-source", "-m/-M/-S" must also be defined.')

    return args

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_do_process(args=None):

    assert args is not None

    # parsing the inputs:
    cfg  = klc.kLocalConfig(
            dataSource = args.dataSource[0],
            acqMode    = args.acqMode[0],
            master     = args.master[0],
            slave      = args.slave[0],
    )

    nb_changes = int(args.nb_changes_misalignment[0])

    # ['T', 'acc_x', 'acc_y', 'acc_z', 'wib_x', 'wib_y', 'wib_z']
    if args.dataSource[0] == klc.kEnumDataSource.SimulatedData.value:
        imu_slave  = kImu(os.path.join(cfg.path_simulated_imu, "imu_leverarm.csv.gz"))
        imu_master = kImu(os.path.join(cfg.path_simulated_imu, "imu_carrier.csv.gz"))
    elif args.dataSource[0] == klc.kEnumDataSource.SassariData.value:
        imu_master = kDbSassari(cfg)['master']
        imu_slave  = kDbSassari(cfg)['slave']

        if False: # just for debugging
            plt.figure(666).clf()
            fig, ax = plt.subplots(1,2,num=666)
            imu_master.plot(ax[0], 'acc')
            imu_master.plot(ax[1], 'wib')

            plt.figure(667).clf()
            fig, ax = plt.subplots(1,2,num=667)
            imu_master.plot(ax[0], 'acc')
            imu_master.plot(ax[1], 'wib')

            plt.show(block=True)

    # sampling period:
    T  = imu_slave.get_time_array()
    Ts = imu_slave.get_sampling_time()

    # max duration:
    if args.tmax[0] is not None:
        T = T[ T < float(args.tmax[0]) ]

    # preparing the deviations along time for the gyro measurements:
    change_at  = np.linspace(0, T.max(), nb_changes+1)
    euler_m2s  = list()

    # from the perspective from the master to slave:
    for i in range(1,nb_changes+1):
        nb_in_range = len(T[ (change_at[i-1] <= T) & (T < change_at[i]) ])

        if args.dataSource[0] == klc.kEnumDataSource.SimulatedData.value:
            euler_random = np.random.randint(-10,11,size=(3,))
        else:
            euler_random = np.zeros(3) # no additional misalignment for the sassari database

        for j in range(nb_in_range):
            euler_m2s.append(euler_random)

    euler_m2s.append(euler_m2s[-1])
    df_euler_m2s = pd.DataFrame(euler_m2s, columns=['phi', 'tta', 'psi'])

    # calculating the euler angles from the perspective of the slave to the master:
    # (this will be used to plot the convergence)
    tmp = list()
    for idx,row in df_euler_m2s.iterrows():
        tmp.append(
                kArrayNav( [row['phi'], row['tta'], row['psi']] ).to_rad().euler2Q().q_inv().Q2euler().to_deg().to_list()
        )

    df_euler_s2m = pd.DataFrame(tmp, columns=['phi', 'tta', 'psi'])

    # The orientation is independent on the lever-arm.
    # The gyro measurements in df_imu_master and df_imu_slave shall be the same, but
    # for the added misaligment in the step before.

    # theta at t=0:
    if False:
        q_s2m_2nd = kArrayNav([5,25,45]).to_rad().euler2Q()
        q_m2s_4th = kArrayNav([-5,-25,-45]).to_rad().euler2Q()
        q_s2m_8th = kArrayNav([10,20,30]).to_rad().euler2Q()
        q_s2m_9th = kArrayNav([-10,-20,30]).to_rad().euler2Q()
    else:
        q_s2m_2nd = kArrayNav([0,0,0]).to_rad().euler2Q()
        q_m2s_4th = kArrayNav([0,0,0]).to_rad().euler2Q()
        q_s2m_8th = kArrayNav([0,0,0]).to_rad().euler2Q()
        q_s2m_9th = kArrayNav([0,0,0]).to_rad().euler2Q()

    # estimation of the lever arm:
    la_8th  = kArrayNav([0,0,0],hvector=0)
    la_9th  = kArrayNav([0,0,0],hvector=0)

    #############################
    ## instances of estimators ##
    #############################

    # Instance 2: beta(t) = [C(q).wib; ||q||^2]
    algo_2nd = kAlgorithm_efol_Cwib_q2(Ts, q_s2m_2nd)

    # Instance 4: alpha = [0(x4); 1.0]; beta = [ (alfa+ - beta-).q; ||q||^2 ]
    algo_4th = kAlgorithm_efol_AplusBminus_q2(Ts, q_m2s_4th)

    # Instance 5: K.q = lbd.q
    algo_5th = kAlgorithm_QMethod(Ts)

    # Instance 7: RLS only for lever-arm
    algo_rls_la = kAlgorithm_rls_leverarm(Ts, kArrayNav([0,0,0], hvector=0))

    # Instance 8: combo(lever-arm, quaternion)
    algo_8th = kAlgorithm_combo(Ts, q_s2m_8th, la_8th)

    # Instance 9: similar to 4th, but with buffer
    algo_9th = kAlgorithm_bufferSamples(Ts, q_s2m_9th, la_9th,  3)

    ###############
    ## time-loop ##
    ###############

    for idx, t in enumerate(tqdm(desc="simulation", total=len(T), ncols=80, iterable=T, mininterval=0.5, ascii=True)):
        wib_master = imu_master.get_wib(idx)
        wib_slave  = imu_slave.get_wib(idx)

        assert not any(np.isnan(wib_master))
        assert not any(np.isnan(wib_slave))

        # applying some deviation to the measurements (aka. misalignment):
        misaligment = kArrayNav(df_euler_m2s.iloc[idx][['phi', 'tta', 'psi']].to_numpy()).to_rad().euler2C()
        wib_slave   = misaligment * wib_slave

        #(==)==(==)==(==)==(==)==(==)==(==)==(==)==(==)==(==)==(==)#

        # calling the efol instances:
        algo_2nd.update( t, wib_master, wib_slave)
        algo_4th.update( t, wib_master, wib_slave)
        algo_5th.update( t, wib_master, wib_slave)
        algo_8th.update( t, wib_master, wib_slave, imu_master.get_acc(idx), imu_slave.get_acc(idx) )
        algo_9th.update( t, wib_master, wib_slave, imu_master.get_acc(idx), imu_slave.get_acc(idx) )

        # test 7 (lever-arm with RLS)
        q_s2m = algo_8th.q_s2m # <= it uses the estimated misalignment from another algorithm
        #q_s2m = algo_4th.q_s2m
        #q_s2m = algo_2nd.q_s2m
        #q_s2m = algo_5th.q_s2m
        algo_rls_la.update(wib_master, imu_master.get_acc(idx), imu_slave.get_acc(idx), q_s2m)

    return {
            'T'                 : T,
            'algo_2nd'          : algo_2nd,
            'algo_4th'          : algo_4th,
            'algo_5th'          : algo_5th,
            'algo_8th'          : algo_8th,
            'algo_9th'          : algo_9th,
            'algo_rls_la'       : algo_rls_la,
            'df_euler_s2m'      : df_euler_s2m,
            'imu_master'        : imu_master,
    }

def fn_show_pictures(process_results = None, args=None):

    assert process_results is not None
    assert args is not None

    # spliting the variables:
    T        = process_results['T']
    algo_5th = process_results['algo_5th']
    algo_8th = process_results['algo_8th']
    algo_9th = process_results['algo_9th']
    algo_rls_la  = process_results['algo_rls_la']
    imu_master   = process_results['imu_master']
    df_euler_s2m = process_results['df_euler_s2m']

    # figures:
    nb_fig = 0

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    nb_fig += 1
    plt.figure(nb_fig).clf()
    fig,ax = plt.subplots(2,1,num=nb_fig,sharex=True)

    # all quaternions together:
    ax[0].plot(algo_8th['T'], algo_8th['quat'], algo_9th['T'], algo_9th['quat'])
    ax[0].grid(True)
    ax[0].set_ylabel('quaternions')

    # quaternions squared norm:
    ax[1].plot(algo_8th['T'], algo_8th['norm'], algo_9th['T'], algo_9th['norm'])
    ax[1].grid(True)
    ax[1].set_ylabel('||Q||^2')

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    nb_fig += 1
    plt.figure(nb_fig).clf()
    fig,ax = plt.subplots(3,1,num=nb_fig,sharex=True)

    # euler angles:
    for i in range(3):
        ax[i].plot(
                algo_5th['T'], algo_5th['euler'][:,i],
                algo_8th['T'], algo_8th['euler'][:,i],
                algo_9th['T'], algo_9th['euler'][:,i],
        )
        ax[i].grid(True)
        ax[i].set_ylabel(f'euler[{i}] [deg]')
        ax[i].set_ylim([-10,10])
        ax[i].legend(('8th', '9th'))

        ax[i].plot(T, df_euler_s2m[['phi', 'tta', 'psi'][i]])

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    for idx, log in enumerate([  algo_8th, algo_9th, ]):
        nb_fig += 1
        plt.figure(nb_fig).clf()
        plt.figure(nb_fig).suptitle(f"err_algo_{['8th', '9th'][idx]}")
        dim_err = len(log['err_unfilt'][0])
        nb_rows = int(sqrt(dim_err))
        nb_cols = ceil(dim_err / nb_rows)
        fig,ax  = plt.subplots(nb_rows, nb_cols, num=nb_fig, sharex=True)
        ax = ax.reshape(-1)


        for i in range(dim_err):
            ax[i].plot( log['T'], log['err_unfilt'][:,i], log['T'], log['err_filt'][:,i],)
            ax[i].grid(True)
            ax[i].legend((f'unfilt[{i}]', f'filt[{i}]',))


    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    for idx, log in enumerate([  algo_8th, algo_9th, ]):
        nb_fig += 1
        plt.figure(nb_fig).clf()
        dim_tta = len(log['dtheta'][0])
        fig,ax = plt.subplots(dim_tta,1,num=nb_fig,sharex=True)
        fig.suptitle(f"d(theta)/dt.algo_{['8th', '9th'][idx]}")

        for i in range(dim_tta):
            ax[i].plot( log['T'], log['dtheta'][:,i] )
            ax[i].grid(True)


    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    for idx, log in enumerate([  algo_8th, algo_9th, ]):
        nb_fig += 1
        plt.figure(nb_fig).clf()
        dim_tta = len(log['dtheta'][0])
        fig,ax = plt.subplots(2,1,num=nb_fig,sharex=True)
        fig.suptitle(f"||d(theta)/dt||.algo_{['8th', '9th'][idx]}")


        for i in range(2):
            ax[i].plot( log['T'], log[ ['norm-dq', 'norm-dla'][i] ])
            ax[i].set_ylabel([ '||d(theta-quat)||', '||d(theta-la)||', ][i])
            ax[i].grid(True)


    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    # selected results (for printing):
    nb_fig += 1
    plt.figure(nb_fig, figsize=(8,11)).clf()
    fig,ax = plt.subplots(4,1,num=nb_fig,sharex=True)

    plot_styles = [
            ('--', 1, 'o'),
            ('-.', 1, 'v'),
            (':', 1, '^'),
            (':', 1, 'D'),
    ]

    for i, attr  in enumerate([ 'euler', 'euler', 'euler', 'norm' ]):
        for j, src in enumerate([ algo_8th, ]):
            if i in {0,1,2}:
                ax[i].plot(T, src[attr][:,i],
                           color='k',
                           linestyle=plot_styles[j][0],
                           linewidth=plot_styles[j][1],
                           marker=plot_styles[j][2],
                           markevery=len(T)//20,
                           markersize=4,
                           markeredgecolor='k',
                           markeredgewidth=1.5,
                           markerfacecolor='none',
                )
                ax[i].set_ylim([-11,11])

            else:
                ax[i].plot(T, src[attr],
                           color='k',
                           linestyle=plot_styles[j][0],
                           linewidth=plot_styles[j][1],
                           marker=plot_styles[j][2],
                           markevery=len(T)//20,
                           markersize=4,
                           markeredgecolor='k',
                           markeredgewidth=1.5,
                           markerfacecolor='none',
                )

            ax[i].grid(True, alpha=0.6)

        if i in {0,1,2}:
            ax[i].plot(T, df_euler_s2m[['phi', 'tta', 'psi'][i]],
                       color='k',
                       linestyle='-',
                       linewidth=1,
            )
        ax[i].set_ylabel(( 'phi [deg]', 'thet [deg]', 'psi [deg]', '||q||^2', )[i])


    ax[0].legend((
        '[C(q).wib; ||q||^2]',
        '[alfa+ - beta-].q; ||q||^2]',
        'eigenvector',
        'combo',
        'groud truth',
    ))
    ax[3].set_xlabel("time [s]")

    # Speichern als Vektor (empfohlen für A4-Druck)
    fig.savefig("figure_a4.pdf")
    fig.savefig("figure_a4.png", dpi=300)

    #===============#
    nb_fig += 1
    plt.figure(nb_fig).clf()
    plt.figure(nb_fig).canvas.manager.set_window_title('la_combo')
    fig,ax = plt.subplots(3,1,num=nb_fig,sharex=True)
    ax = ax.reshape(-1)

    for i in range(3):
        ax[i].plot(T, [j[i] for j in algo_8th['la']])
        ax[i].grid(True, alpha=0.5)
        ax[i].set_ylabel(f'la_combo[{i}]')

    #===============#
    nb_fig += 1
    plt.figure(nb_fig).clf()
    plt.figure(nb_fig).canvas.manager.set_window_title('la_rls')
    fig,ax = plt.subplots(4,1,num=nb_fig,sharex=True)
    ax = ax.reshape(-1)

    for i in range(3):
        ax[i].plot(T, algo_rls_la[i])
        ax[i].grid(True, alpha=0.5)
        ax[i].set_ylabel(f'unfilt-error[{i}]')

    ax[3].plot(T, algo_rls_la["la"])
    ax[3].grid(True, alpha=0.5)
    ax[3].set_ylabel('la_rls')

    #===============#
    nb_fig += 1
    plt.figure(nb_fig).clf()
    plt.figure(nb_fig).canvas.manager.set_window_title('lever-arm')
    fig,ax = plt.subplots(3,1,num=nb_fig,sharex=True)
    ax = ax.reshape(-1)

    for i in range(3):
        ax[i].plot(
                algo_8th['T'], [j[i] for j in algo_8th['la']],
                algo_9th['T'], [j[i] for j in algo_9th['la']],
                T, algo_rls_la["la"][:,i],
        )
        ax[i].grid(True, alpha=0.5)
        ax[i].set_ylabel(f'[{["X [m]", "Y [m]", "Z [m]"][i]}]')
        ax[i].legend(( 'combo', 'combo-buffer', 'rls' ))

    #===============#
    if args.dataSource[0] == klc.kEnumDataSource.SassariData.value:
        plt.figure(100).clf()
        plt.figure(100).canvas.manager.set_window_title('imu-sassari')
        fig, ax = plt.subplots(2,1,num=100,sharex=True)
        imu_master.plot(ax[0], "acc")
        imu_master.plot(ax[1], "wib")



    #===============#
    for idx, log in enumerate([algo_8th, algo_9th]):
        nb_fig += 1
        plt.figure(nb_fig).clf()
        plt.figure(nb_fig).canvas.manager.set_window_title('norm-betas')
        plt.figure(nb_fig).suptitle(f"{['algo_8th', 'algo_9th'][idx]}: norm(beta{1,2,3})")
        fig,ax = plt.subplots(3,1,num=nb_fig,sharex=True)
        ax = ax.reshape(-1)

        for i in range(3):
            ax[i].plot(
                    log['T'], log['mask_unfilt'][:,i],
                    log['T'], log['mask_filt'][:,i]
            )
            ax[i].grid(True, alpha=0.5)
            ax[i].set_ylabel([ '||beta_1||', '||beta_2||', '||beta_3||', ][i])

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    # export PDF file:
    if args.pdffile[0] is not None:
        kPdfUtils.export_figures_2_pdf(args.pdffile[0])
        print(f"--> PDF file with all pictures created: '{args.pdffile[0]}'")

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    for fig in plt.get_fignums():
        plt.figure(fig).canvas.flush_events()
        plt.figure(fig).canvas.draw()

    plt.show(block=args.is_block)
    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
if __name__ == "__main__":
    args = fn_parsing()
    ret  = fn_do_process(args=args)
    fn_show_pictures(ret, args=args)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
