#TODO: cut the first frames, and split to frames
import glob
import sys
import os

os.system('mkdir frames')

fp_list = glob.glob('./vid-cut/*.mp4')
for fp in fp_list:
    basename = os.path.basename(fp)#file name no path
    filename, ext = os.path.splitext(basename)#file name no path and extension

    ##createa a folder in frames with name is filename no ext
    os.system(f"mkdir ./frames/{filename}")
    
    ## split frames, put into ./frames/{filename} folder
    cmd2 = f'ffmpeg -i ./vid-cut/{basename} ./frames/{filename}/%04d.jpg -hide_banner'
    print (f"executing: {cmd2}")
    os.system(cmd2)