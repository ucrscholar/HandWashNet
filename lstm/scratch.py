import os
import GPUtil
from ilab.utils import Notify

from tensorflow_core.python import expand_dims
#TODO change the device
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
while(len(deviceIDs)==-1):
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.5, maxMemory=0.2, includeNan=False, excludeID=[],
                                    excludeUUID=[])
Notify(0)
deviceIDs.append(1);
os.system(" ~/anaconda3/bin/python3.7 mainConvLSTM.py {}".format(deviceIDs[0]))