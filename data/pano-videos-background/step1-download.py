import os
import glob
import sys

#################fp is the a single file name containting list of youtube url to be downloaded######
fp = sys.argv[1]
fp0_list = [fp0 for fp0 in glob.glob('./*txt') if os.path.basename(fp0) != fp ]

print ('beginging parsing list of videos') 
print (f'file to download : {fp}')
print (f'file of downloaded vids: {fp0_list}')


##################get vid_list is the list of file to be downloaded, and vid0_list is the list of downloaded videos#####
vid_list = set(open(fp).read().split('\n')[:-1])
vid0_list = set()
for fp0 in fp0_list:
        vid0_list.union(set(open(fp0).read().split('\n')[1:-1]))

##################make sure we don't download duplicated video, by removing vids in vid_list that are also in vid0_list###
vid_list = vid_list - vid0_list


##################download all new videos    ####################
for vid in vid_list:
    print (f'downloading: {vid}')
    os.system(f"yt-dlp -f worst[ext=mp4] --no-check-certificate --user-agent \'\' {vid}")