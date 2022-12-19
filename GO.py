import shlex
import datetime
from time import sleep
from subprocess import Popen, PIPE
import sys
import os

time = datetime.datetime.now()
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 1
while True:
    print(time)
    sleep(2)
    print(os.getcwd())
    os.execv("python GO.py", sys.argv)
    if False:
        c = f'pkill -f "GO.py" ; python3 GO.py {N} >> GO.log'
        #Popen(shlex.split(c))
        Popen(c, shell=True)
