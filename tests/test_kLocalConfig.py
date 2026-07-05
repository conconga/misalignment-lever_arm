#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
import sys
print( "**************************************" )
print(f"** __name__    = {__name__}")
print(f"** __package__ = {__package__}")
print(f"** sys.path[0] = {sys.path[0]}")


import pytest
from kLocalConfig import (
        kLocalConfig,
        kEnumDataSource,
        kEnumAcquisitionMode,
        kEnumSensorName,
)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

def fn_set_trace():
    import pudb
    pudb.set_trace()

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class TestClass_kLocalConfig:

    def test_emptyConstructor(self):
        with pytest.raises(AssertionError):
            cfg = kLocalConfig()

    def test_sissara_with_typo(self):
        with pytest.raises(AssertionError):
            cfg = kLocalConfig(dataSource = "not a valid dataSource")

    def test_sissara_with_no_acqMode(self):
        with pytest.raises(AssertionError):
            cfg = kLocalConfig(dataSource = "sissara")

    def test_full_cfg(self):
        cfg = kLocalConfig(dataSource="sassari", acqMode="slow", master="xs1", slave="ap1")
        assert cfg.dataSource == "sassari"
        assert cfg.acqMode    == "slow"
        assert cfg.master     == "xs1"
        assert cfg.slave      == "ap1"

    def test_sassari_fast_xs1_sh1_filename(self):
        cfg = kLocalConfig(dataSource="sassari", acqMode="fast", master="xs1", slave="sh1")

        name = cfg['master']
        for i in ['data', 'imu', 'xs1', 'fast', 'csv']:
            assert i in name
        assert "sh1" not in name

        name = cfg['slave']
        for i in ['data', 'imu', 'sh1', 'fast', 'csv']:
            assert i in name
        assert "xs1" not in name

        assert cfg['gt'] is None

    def test_simulated_filename(self):
        cfg = kLocalConfig(dataSource="simulated")

        name = cfg['master']
        for i in ['data', 'imu', 'carrier', 'csv']:
            assert i in name

        name = cfg['slave']
        for i in ['data', 'imu', 'leverarm', 'csv']:
            assert i in name

        name = cfg['gt']
        for i in ['data', 'gt', 'csv']:
            assert i in name

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
