#########################################################################
# File Name: run.sh
# Author: yanxuecan
# Mail: yanxuecan.iron@bytedance.com
# Created Time: Mon Jul 10 17:50:30 2017
#########################################################################
#!/bin/bash

python gpucb.py
rm output.gif
ffmpeg -framerate 3 -i fig_%02d.png output.gif
rm *.png
