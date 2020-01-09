import datetime
import os
import time

import GPUtil
from ilab.utils import Notify

from tensorflow_core.python import expand_dims
#TODO change the device
val = -1;
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
if(len(deviceIDs) !=0):
    val = deviceIDs[0]
while(val !=0):
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    if (len(deviceIDs) != 0):
        val = deviceIDs[0]
    time.sleep(1)
    print("{}".format(datetime.datetime.now().time()))
Notify(0)

os.system(" ~/gogogo MainconvLSTMCom.py")