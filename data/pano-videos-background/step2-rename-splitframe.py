import glob
import os
import sys

#TODO: move all downloaded files from vid-raw to vid-renamed
fp0_list = sorted(glob.glob('vid-raw/*.mp4'))
fp1_list = sorted(glob.glob('vid-cut/*.mp4'))

#########################check the id of the files in vid-rename folder

if len(fp1_list) == 0:
    idx0 = 0
else:
    #get the max id of the video
    vid_list = []
    for fp1 in fp1_list:
        filepath, ext = os.path.splitext(fp1)
        filename = filepath.split('/')[-1]
        print (fp1, filepath, filename)
        idxitem = int(filename.replace('.mp4', '').replace('vid', ''))#we assume file name is vid-{idx}, so split '_' gives us the id
        vid_list.append(idxitem)
        #print (fp1, filename, idxitem)
    vid_list = sorted(vid_list)
    idx0 = vid_list[-1]
    #print (vid_list)
print (f'changing file name of downloaded video, starting at {idx0}')

########################move all raw files from vid-raw to vid-rename####

########move raw files from vid-raw to vid-renamed
file = open('./filename-mapping.txt', 'a')
for fp in fp0_list:
    filename, ext = os.path.splitext(fp)
    dfname = f'./vid-renamed/vid{idx0}{ext}'
    print (f"cp \'{fp}\' {dfname}")
    os.system(f"cp \"{fp}\" {dfname}")
    file.write(f'{fp}\t{dfname}\n')
    idx0 += 1
