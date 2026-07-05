#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import sys
print( "**************************************" )
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")

import numpy as np
from types import SimpleNamespace
from sc_misalign_leverarm_estimator import fn_do_process, fn_show_pictures

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class TestClass_Simu_ScriptAlgos:

    def test_main_simulated(self):
        args = {
                'dataSource': ['simulated'],
                'acqMode'   : [None],
                'master'    : [None],
                'slave'     : [None],
                'nb_changes_misalignment' : [1],
                'is_block'  : False,
                'tmax'      : ['3'],
                'pdffile'   : [None],
        }

        args = SimpleNamespace(**args)
        ret  = fn_do_process(args=args)
        fn_show_pictures(ret, args) # <= to test whether ret carries all necessary content

    def get_random_acqMode(self):
        collection = [ 'fast', 'medium', 'slow', ]
        return collection[ np.random.randint(0, len(collection)) ]

    def get_random_sensor(self):
        collection = [ 'xs1', 'xs2', 'ap1', 'ap2', 'sh1', 'sh2', ]
        return collection[ np.random.randint(0, len(collection)) ].upper()

    def test_random(self):
        lst_acqMode = list()
        lst_sensor  = list()
        for i in range(1000):
            lst_acqMode.append(self.get_random_acqMode())
            lst_sensor.append(self.get_random_sensor())

        assert len(set(lst_acqMode)) == 3
        assert len(set(lst_sensor)) == 6

    def test_main_sassari(self):
        args = {
                'dataSource': ['sassari'],
                'acqMode'   : [self.get_random_acqMode()],
                'master'    : [self.get_random_sensor()],
                'slave'     : [self.get_random_sensor()],
                'nb_changes_misalignment' : [1],
                'is_block'  : False,
                'tmax'      : ['3'],
                'pdffile'   : [None],
        }

        args = SimpleNamespace(**args)
        ret  = fn_do_process(args=args)
        fn_show_pictures(ret, args) # <= to test whether ret carries all necessary content

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
