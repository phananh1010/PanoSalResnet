TARGET_IMG_H = 144
TARGET_IMG_W = 256
TARGET_SAL_H = 45
TARGET_SAL_W = 80
TARGET_STANDARD_SAL_H = 90
TARGET_STANDARD_SAL_W = 180
TARGET_GAUSSIAN_BLUR = 19

fps_dict = {'paris': 59.94, 'roller': 29, 'timelapse': 30, 'venise': 25, 'diving': 29, \
           '0': 29.97, '1': 29.97, '2': 30, '3': 29.97, '4': 29.97, \
           '5': 29.97, '6': 25, '7': 25, '8': 29.97,\
           'coaster': 29.97, 'coaster2': 60, 'diving2': 30, 'drive': 30, 'game': 30, 'landscape': 30, \
           'pacman': 25, 'panel': 23.98, 'ride': 24, 'sport': 29.97
           }

fps_dict = {1:{'paris': 59.94, 'roller': 29, 'timelapse': 30, 'venise': 25, 'diving': 29},
           2:{'0': 29.97, '1': 29.97, '2': 30, '3': 29.97, '4': 29.97, \
           '5part1': 29.97, '6part1': 25, '7': 25, '8': 29.97},
           3:{'coaster': 29.97, 'coaster2': 60, 'diving2': 30, 'drive': 30, 'game': 30, 'landscape': 30, \
           'pacman': 25, 'panel': 23.98, 'ride': 24, 'sport': 29.97}}

topic_dict = {1: ['paris', 'roller', 'venise', 'diving', 'timelapse'],
              2: ['0', '1', '2', '3', '4', '5part1', '6part1', '7', '8'], #remove 5 & 6 since those are stored in multiple files
              3: ['coaster2', 'diving', 'drive', 'game', 'landscape', 'pacman', 'ride', 'sport']}
