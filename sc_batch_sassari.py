#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

import os
import cv2
import numpy  as np
import argparse
import matplotlib.pyplot as plt
import multiprocessing   as mp

from matplotlib.patches import Patch
from types       import SimpleNamespace

from kPdfUtils   import kPdfUtils

from submodules.gcnutils.knavigation import kArrayNav
from sc_misalign_leverarm_estimator  import fn_do_process as oneshotprocess
from fn_cv2_contrast                 import fn_apply_brightness_contrast

import kLocalConfig       as klc

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

def fn_parsing():

    ##########################
    ## parser configuration ##
    ##########################

    parser = argparse.ArgumentParser(
            description="Analysis of misalignment based on the Sassari dataset",
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
            '--export-assets',
            required=False,
            action="store_true",
            dest='is_export_assets',
            default=False,
            help='Enables export of assets (svg files) before leaving.',
    )

    parser.add_argument(
            '--no-mp',
            required=False,
            action="store_true",
            dest='is_no_multiprocessing',
            default=False,
            help='Turns on/off multiprocessing of batch tests',
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
    print(". Parsing arguments:")
    print("args = ", args)
    print()

    ###########################################
    ## fixing pdf filename without extension ##
    ###########################################
    if args.pdffile[0] is not None:
        args.pdffile[0] = kPdfUtils.fix_filename_extension(args.pdffile[0])

    return args

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_get_acqMode(idx):
    collection = [ 'fast', 'medium', 'slow', ]
    return collection[idx]

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_get_sensor(idx):
    collection = [ 'xs1', 'xs2', 'ap1', 'ap2', 'sh1', 'sh2', ]
    return collection[idx].upper()

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_get_testname(acqMode, sensorA, sensorB):
    return f'{acqMode}_{sensorA}_{sensorB}'

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def gen_acqMode(from_idx=0):
    for i in range(from_idx, 3):
        yield i, fn_get_acqMode(i)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def gen_sensor(from_idx=0, but_not_idx=None):
    for i in range(from_idx, 6):
        if but_not_idx is not None:
            if i == but_not_idx:
                continue
        yield i, fn_get_sensor(i)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kOneShotProcess():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, testname):
        print(f"--> starting kOneShotProcess() with pid = {os.getpid()}..")
        ret_oneshot = oneshotprocess(self.cfg)
        ret_summary = fn_summarize(ret_oneshot)
        print(f"--> finished kOneShotProcess() with pid = {os.getpid()}..")

        return testname, { 'full': ret_oneshot, 'summary': ret_summary, }

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_multipleshotprocess(args=None):

    assert args is not None

    result       = dict()
    asyncresults = list()

    with mp.Pool() as pool:
        for idx_acqMode, acqMode in gen_acqMode():
            for idx_sensorA, sensorA in gen_sensor():
                for idx_sensorB, sensorB in gen_sensor(from_idx=idx_sensorA+1):

                    tmp_args = {
                            'dataSource': ['sassari'],
                            'acqMode'   : [acqMode],
                            'master'    : [sensorA],
                            'slave'     : [sensorB],
                            'nb_changes_misalignment' : ['1'], # <= it doesn't matter
                            'is_block'  : args.is_block,
                            'tmax'      : [args.tmax[0]],
                    }

                    tmp_args = SimpleNamespace(**tmp_args)
                    testname = fn_get_testname(acqMode, sensorA, sensorB)

                    if True:
                        processOneShot = kOneShotProcess(tmp_args)
                        asyncresult    = pool.apply_async(processOneShot.run, args=(testname, ))

                        if args.is_no_multiprocessing:
                            asyncresult.wait(3*60)

                        asyncresults.append(asyncresult)
                    else:
                        ret_oneshot = oneshotprocess(tmp_args)
                        ret_summary = fn_summarize(ret_oneshot)
                        result[testname] = { 'full': ret_oneshot, 'summary': ret_summary, }

        # collect all results:
        for asyncresult in asyncresults:
            if not asyncresult.wait(3*60):
                # it gets in here only if there was no timeout
                ret = asyncresult.get()
                result[ret[0]] = ret[1]

    return result

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_summarize(ret_oneshot):
    summary        = dict()
    nb_samples_std = 20

    try:
        # algo_2nd => only misaligment:
        summary['2nd.last_euler_s2m'] = kArrayNav(ret_oneshot['algo_2nd']['q_s2m'][-1]).q_norm().Q2euler().to_deg().to_list()
        summary['2nd.std_euler_s2m'] = [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in  ret_oneshot['algo_2nd']['q_s2m'][-nb_samples_std:]]
        summary['2nd.std_euler_s2m'] = np.asarray(summary['2nd.std_euler_s2m']).std(axis=0).tolist()
    except Exception:
        pass

    try:
        # algo_4th => only misalignment:
        summary['4th.last_euler_s2m'] = kArrayNav(ret_oneshot['algo_4th']['q_s2m'][-1]).q_norm().Q2euler().to_deg().to_list()
        summary['4th.std_euler_s2m'] = [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in  ret_oneshot['algo_4th']['q_s2m'][-nb_samples_std:]]
        summary['4th.std_euler_s2m'] = np.asarray(summary['4th.std_euler_s2m']).std(axis=0).tolist()
    except Exception:
        pass

    try:
        # algo_5th => only misalignment:
        summary['5th.last_euler_s2m'] = kArrayNav(ret_oneshot['algo_5th']['q_s2m'][-1]).q_norm().Q2euler().to_deg().to_list()
        summary['5th.std_euler_s2m'] = [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in  ret_oneshot['algo_5th']['q_s2m'][-nb_samples_std:]]
        summary['5th.std_euler_s2m'] = np.asarray(summary['5th.std_euler_s2m']).std(axis=0).tolist()
    except Exception:
        pass

    try:
        # algo_rls_la => only Lever-Arm:
        summary['6th.LA.last'] = ret_oneshot['algo_rls_la']['la'][-1].tolist()
        summary['6th.LA.std'] = ret_oneshot['algo_rls_la']['la'][-nb_samples_std:].std(axis=0).tolist()
    except Exception:
        pass

    try:
        # algo_8th [combo] => both misalignment and Lever-Arm:
        summary['8th.last_euler_s2m'] = kArrayNav(ret_oneshot['algo_8th']['q_s2m'][-1]).q_norm().Q2euler().to_deg().to_list()
        summary['8th.std_euler_s2m'] = [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in  ret_oneshot['algo_8th']['q_s2m'][-nb_samples_std:]]
        summary['8th.std_euler_s2m'] = np.asarray(summary['8th.std_euler_s2m']).std(axis=0).tolist()
    except Exception:
        pass

    try:
        summary['8th.LA.last'] = ret_oneshot['algo_8th']['la'][-1].tolist()
        summary['8th.LA.std'] = ret_oneshot['algo_8th']['la'][-nb_samples_std:].std(axis=0).tolist()
    except Exception:
        pass

    try:
        # algo_9th [combo] => both misalignment and Lever-Arm:
        summary['9th.last_euler_s2m'] = kArrayNav(ret_oneshot['algo_9th']['q_s2m'][-1]).q_norm().Q2euler().to_deg().to_list()
        summary['9th.std_euler_s2m'] = [kArrayNav(i).q_norm().Q2euler().to_deg().to_list() for i in  ret_oneshot['algo_9th']['q_s2m'][-nb_samples_std:]]
        summary['9th.std_euler_s2m'] = np.asarray(summary['9th.std_euler_s2m']).std(axis=0).tolist()
    except Exception:
        pass

    try:
        summary['9th.LA.last'] = ret_oneshot['algo_9th']['la'][-1].tolist()
        summary['9th.LA.std'] = ret_oneshot['algo_9th']['la'][-nb_samples_std:].std(axis=0).tolist()
    except Exception:
        pass

    return summary

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_get_next_figure(nb_fig, nb_rows, nb_cols):
    plt.figure(nb_fig, figsize=(12,8)).clf()
    fig,ax = plt.subplots(nb_rows, nb_cols, num=nb_fig, sharex=True)
    if hasattr(ax, "__iter__"):
        ax = ax.reshape(-1)
    return ax

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_get_color(algo):
    if algo == "2nd":
        return "#ff0000"
    elif algo == "4th":
        return "#00ff00"
    elif algo == "5th":
        return "#0000ff"
    elif algo == "6th":
        return "#ffff00"
    elif algo == "8th":
        return "#00ffff"
    elif algo == "9th":
        return "#ff00ff"
    else:
        return "#000000"

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_hex_to_bgr(hex_str: str):
    r, g, b = fn_hex_to_rgb(hex_str)
    return (b, g, r)  # OpenCV erwartet BGR

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_make_a_picture( ax, sensorA, lst_points ):
    """
    lst_points = [     # (with RGB!)
            ((0,   0), "#ff0000"),
            ((30, 10), "#00ff00"),
            ((60, 30), "#0000ff"),
    ]

    **  Points aligned with X-north/up and Z-down, in [mm] **

    """

    # origin of the sensors:
    origin = {
            'xs1': (560, 560),
            'xs2': (387, 560),
            'ap1': (595, 462),
            'ap2': (430, 464),
            'sh1': (584, 286),
            'sh2': (389, 290),
    }

    # scale (with FreeCAD):
    Sx = 455.32/902   # [mm/px]
    Sy = 567.89/1125  # [mm/px]

    # origin of "this" sensor:
    zero = origin[sensorA.lower()]

    # load BGR:
    img = cv2.imread(os.path.join(
        klc.kLocalConfig.path_sassari_imu,
        "1619619646-3053265616.png"
    ))

    # cropping:
    crop0 = (240,180)
    crop1 = (740,715)
    img   = img[crop0[1]:crop1[1], crop0[0]:crop1[0]] # [rows,cols]
    zero  = (zero[0]-crop0[0], zero[1]-crop0[1])

    # adjust contrast:
    img = fn_apply_brightness_contrast(img, 80, -30)

    # convert to RGB to show with matplotlib:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # just for testing:
    #lst_points = [
    #        ((0,0), "#00ff00"),
    #        ((40,40), "#ffffff"),
    #        ((10,-50), "#ff0000"),
    #]

    for point, color_str in lst_points:
        x = zero[0] + int(point[1] / Sx)
        y = zero[1] - int(point[0] / Sy)
        print(f'point = ({point[0]:.1f}, {point[1]:.1f})[mm] = ({x}, {y})[px] with color "{color_str}"')

        # circle at origin:
        radius = 6
        color  = fn_hex_to_rgb(color_str)
        cv2.circle(img, (x,y), radius, color, thickness=-1)

        # line:
        cv2.line(img, zero, (x,y), (0,0,0), 3)
        cv2.line(img, zero, (x,y), color,   2)


    ax.imshow(img)


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
def fn_show_pictures(process_results = None, args=None):

    assert process_results is not None
    assert args is not None

    # figures:
    nb_fig = 0

    nb_fig += 1
    plt.figure(nb_fig, figsize=(12,8)).clf()
    fig_la, ax = plt.subplots(2,3,num=nb_fig)
    ax_la = ax.reshape(-1)
    #ax_la = fn_get_next_figure(nb_fig, 2,3)
    plt.figure(nb_fig).canvas.manager.set_window_title("LA")
    plt.figure(nb_fig).suptitle("LA")

    for idx_sensorA, sensorA in gen_sensor():
        lst_la    = list()

        for idx_sensorB, sensorB in gen_sensor(but_not_idx=idx_sensorA):
            lst_euler = list()

            for idx_acqMode, acqMode in gen_acqMode():

                # name of the test is not comutative:
                if idx_sensorA < idx_sensorB:
                    testname = fn_get_testname(acqMode, sensorA, sensorB)
                    sign     = 1.0
                else:
                    testname = fn_get_testname(acqMode, sensorB, sensorA)
                    sign     = -1.0

                # collect full history of euler calculation:
                lst_euler.append( (process_results[testname]['full']['algo_2nd']['T'], process_results[testname]['full']['algo_2nd']['euler'], fn_get_color("2nd")) )
                lst_euler.append( (process_results[testname]['full']['algo_4th']['T'], process_results[testname]['full']['algo_4th']['euler'], fn_get_color("4th")) )
                lst_euler.append( (process_results[testname]['full']['algo_5th']['T'], process_results[testname]['full']['algo_5th']['euler'], fn_get_color("5th")) )
                lst_euler.append( (process_results[testname]['full']['algo_8th']['T'], process_results[testname]['full']['algo_8th']['euler'], fn_get_color("8th")) )
                lst_euler.append( (process_results[testname]['full']['algo_9th']['T'], process_results[testname]['full']['algo_9th']['euler'], fn_get_color("9th")) )

                # collect last calculated sample of lever-arm:
                temp = process_results[testname]['summary']
                for algo in [i for i in list(temp) if ("LA.last" in i)]:
                    lst_la.append( ([sign * i for i in temp[algo]], fn_get_color(algo[:3]), f"{sensorA}-{sensorB}-{acqMode}-{algo[:3]}") )

            # here, lst_euler contains all full-history of each algorithm that calculates the
            # misalignment between sensorA and sensorB, over all acqModes (fast/slow/medium).
            nb_fig += 1
            ax = fn_get_next_figure(nb_fig, 3,1)
            plt.figure(nb_fig).canvas.manager.set_window_title(f"euler({sensorA},{sensorB})")
            plt.figure(nb_fig).suptitle(f"euler({sensorA},{sensorB})")
            for i in lst_euler: # nb_algos x fast/medium/slow
                for j in range(3):
                    if i[1].size > 0:
                        ax[j].plot(i[0], i[1][:,j], color=i[2])

            for j in range(3):
                ax[j].grid(True, alpha=0.5)
                ax[j].set_ylabel(f'{["phi", "theta", "psi"][j]}')
                ax[j].set_ylim((-2,2))

            if args.is_export_assets and (sensorA.lower() == "xs2") and (sensorB.lower() == "sh1"):
                plt.figure(nb_fig).tight_layout()
                plt.figure(nb_fig).savefig(os.path.join(klc.kLocalConfig.path_assets, "xs2-sh1.svg"), transparent=True)


        # And here the lst_la contains all lever-arm calculated from sensorA to any other;
        # all of them will be depicted in the same figure.
        # The way to understand the next pictures is like this:
        #  - sitting on the sensorA, where are the other 5 sensors?
        #  - the origin of the current sensor (sensorA) is always (0,0)

        # draw the lines on the background:
        fn_make_a_picture( ax_la[idx_sensorA], sensorA,
                          [((x*1e3, y*1e3), color) for (x,y,z),color,debug in lst_la], # <== from [m] to [mm]
        )

        # convert each point to a line:
        for i in lst_la:
            point, color, debug = i
            ax_la[idx_sensorA].plot([0, point[0]], [0, point[1]], color=color)


        ax_la[idx_sensorA].set_ylabel(f'from {sensorA}')
        ax_la[idx_sensorA].grid(True, alpha=0.5)
        #ax_la[idx_sensorA].invert_xaxis()
    fig_la.tight_layout()

    # asset:
    if args.is_export_assets:
        fig_la.savefig(os.path.join(klc.kLocalConfig.path_assets, "lever-arm.svg"), transparent=True)

    #//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\--//==\\#
    # simply, a legend
    nb_fig += 1
    plt.figure(nb_fig, figsize=(2,2)).clf()
    ax = fn_get_next_figure(nb_fig, 1,1)
    plt.figure(nb_fig).canvas.manager.set_window_title("legend")
    plt.figure(nb_fig).suptitle("legend")

    labels  = [
            ('2nd', 'algo_2nd'),
            ('4th', 'algo_4th'),
            ('5th', 'q-method'),
            ('6th', 'RLS'),
            ('8th', 'combo'),
            ('9th', 'combo-buffer'),
    ]

    handles = [
        Patch(facecolor=fn_get_color(i), edgecolor=fn_get_color(i), label=j)
        for i,j in labels
    ]

    ax.legend(handles=handles, loc='upper left')
    ax.axis('off')

    # asset:
    if args.is_export_assets:
        plt.figure(nb_fig).tight_layout()
        plt.figure(nb_fig).savefig(os.path.join(klc.kLocalConfig.path_assets, "legend.svg"), transparent=True)

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

    ret  = fn_multipleshotprocess(args=args)

    fn_show_pictures(ret, args=args)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
