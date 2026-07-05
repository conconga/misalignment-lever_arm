#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import sys
print( "**************************************" )
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")

import numpy as np
from types import SimpleNamespace
import sc_batch_sassari as bs

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class TestClass_BatchProcess:

    def test_SassariBatchProcess_1second(self):
        args = {
                'tmax'                  : ['0.5'],
                'is_block'              : False,
                'pdffile'               : [None],
                'is_no_multiprocessing' : False,
                'is_export_assets'      : False,
        }

        args = SimpleNamespace(**args)
        ret  = bs.fn_multipleshotprocess(args=args)

        # to test whether ret carries all necessary content
        # also whether all dicts for the pictures are available and correct
        bs.fn_show_pictures(ret, args)

    def test_generator_sensor(self):
        A = list()

        for i,j in bs.gen_sensor():
            A.append(i)

        assert A == list(range(6))
        
    def test_generator_sensor_from(self):
        A = list()

        for i,j in bs.gen_sensor(from_idx=3):
            A.append(i)

        assert A == [3,4,5]

    def test_generator_sensor_but_not(self):
        A = list()

        for i,j in bs.gen_sensor(but_not_idx=4):
            A.append(i)

        assert A == [0,1,2,3,5]

    def test_generator_sensor_from_but(self):
        A = list()

        for i,j in bs.gen_sensor(from_idx=3, but_not_idx=4):
            A.append(i)

        assert A == [3,5]

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
