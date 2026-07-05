#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
import os
from enum import Enum
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#

class kEnumDataSource (Enum):
    SimulatedData = "simulated"
    SassariData   = "sassari"

class kEnumAcquisitionMode (Enum):
    Slow   = "slow"
    Medium = "medium"
    Fast   = "fast"

class kEnumSensorName (Enum):
    # Xsens IMU:
    XS1 = "xs1"
    XS2 = "xs2"

    # APDM - Opal:
    AP1 = "ap1"
    AP2 = "ap2"

    # Shimmer - Shimmer3:
    SH1 = "sh1"
    SH2 = "sh2"


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
#                                                                                  #
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
class kLocalConfig:

    # hardcoded:
    path_simulated_imu = "./data/"
    path_sassari_imu   = "./data/sassari_dataset/mimu_optical_dataset_caruso_sassari-5.0/"
    path_assets        = "./assets"

    def __init__(self,
                 dataSource = None, # in kEnumDataSource
                 acqMode    = None, # in kEnumAcquisitionMode
                 master     = None, # in kEnumSensorName
                 slave      = None, # in kEnumSensorName
    ):

        assert dataSource is not None
        assert dataSource.lower() in kEnumDataSource, f"dataSource = {str(dataSource)} is not a valid entry"
        self.dataSource = dataSource.lower()

        if dataSource == kEnumDataSource.SassariData.value:
            assert acqMode.lower() in kEnumAcquisitionMode, f"acqMode = {str(acqMode)} is not a valid entry"
            assert master.lower()  in kEnumSensorName, f"master = {str(master)} is not a valid entry"
            assert slave.lower()   in kEnumSensorName, f"slave = {str(slave)} is not a valid entry"

            self.acqMode = acqMode.lower()
            self.master  = master.lower()
            self.slave   = slave.lower()

    def __getitem__(self, idx):
        if idx == "master":
            if self.dataSource == kEnumDataSource.SimulatedData.value:
                master = os.path.join( self.path_simulated_imu, "imu_carrier.csv.gz" )
            elif self.dataSource == kEnumDataSource.SassariData.value:
                master = os.path.join( self.path_sassari_imu, f"imu_{self.acqMode}_{self.master}.csv.gz" )

            return master

        elif idx == "slave":
            if self.dataSource == kEnumDataSource.SimulatedData.value:
                slave = os.path.join( self.path_simulated_imu, "imu_leverarm.csv.gz" )
            elif self.dataSource == kEnumDataSource.SassariData.value:
                slave = os.path.join( self.path_sassari_imu, f"imu_{self.acqMode}_{self.slave}.csv.gz" )

            return slave

        elif idx == "gt": # "ground truth"
            if self.dataSource == kEnumDataSource.SimulatedData.value:
                gt = os.path.join( self.path_simulated_imu, "gt_carrier.csv.gz" )
            elif self.dataSource == kEnumDataSource.SassariData.value:
                gt = None

            return gt

        elif idx == "mat": # to retrieve the name of the mat file in the sassari database
            assert self.dataSource == kEnumDataSource.SassariData.value

            return os.path.join(self.path_sassari_imu, f"{self.acqMode}_v4.mat")


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>#
