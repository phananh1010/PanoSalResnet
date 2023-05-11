import glob
import os
import sys

#TODO: move all downloaded files from vid-raw to vid-renamed
fp0_list = sorted(glob.glob('vid-raw/*'))
fp1_list = sorted(glob.glob('vid-renamed/*'))

#########################check the id of the files in vid-rename folder

if len(fp1_list) == 0:
    idx0 = 0
else:
    filename, ext = os.path.splitext(fp1_list[-1])
    idx0 = int(filename.split('-')[-1])#we assume file name is vid-{idx}, so split '_' gives us the id
    
print ('changing file name of downloaded video, starting at 0')

########################move all raw files from vid-raw to vid-rename####

#move raw files from vid-raw to vid-renamed
file = open('./filename-mapping.txt', 'a')
for fp in fp0_list:
    filename, ext = os.path.splitext(fp)
    dfname = f'./vid-renamed/vid-{idx0}{ext}'
    print (f"cp \'{fp}\' {dfname}")
    os.system(f"cp \"{fp}\" {dfname}")
    file.write(f'{fp}\t{dfname}\n')
    idx0 += 1
