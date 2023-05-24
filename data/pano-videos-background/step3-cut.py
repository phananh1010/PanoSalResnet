#TODO: cut the first frames, and split to frames
import glob
import sys
import os


fp_list = glob.glob('./vid-renamed/*.mp4')
for fp in fp_list:
    basename = os.path.basename(fp)#file name no path
    filename, ext = os.path.splitext(basename)#file name no path and extension
    
    #TODO: cut first one minutes of the files 
    cmd = f'yes | /usr/bin/ffmpeg -ss 00:00:00.0 -i {fp} -vf \"fps=fps=30\" -t 00:01:00.0 ./vid-cut/{basename}'
    print (f"executing: {cmd}")
    os.system(cmd)
    
    ##createa a folder in frames with name is filename no ext
    os.system(f"mkdir ./frames/{filename}")
    
    ## split frames, put into ./frames/{filename} folder
    cmd2 = f'/usr/bin/ffmpeg -i ./vid-cut/{basename} ./frames/{filename}/%04d.jpg -hide_banner'
    print (f"executing: {cmd2}")
    os.system(cmd2)