import pickle

import saldat_head_orientation
import header_headoren_salcorr as header


dirpath1 = header.dirpath1#u'./data/head-orientation/dataset1'
dirpath2 = header.dirpath2#u'./data/head-orientation/dataset2/Experiment_1'
dirpath3 = header.dirpath3#u'./data/head-orientation/dataset3/sensory/orientation'
ext1 = header.ext1
ext2 = header.ext2
ext3 = header.ext3
headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, dirpath3, ext1, ext2, ext3)

SAL_PATH = './dataset/pano-saliency'

def f_create_key(ds, topic):
    return '{}_{}'.format(ds, topic)

def f_create_log_key(dir_path, prefix, f_ds, f_topic, s_ds, s_topic, postfix):
    return '{}{}_{}_{}_{}_{}_{}'.format(dir_path, prefix, f_ds, f_topic, s_ds, s_topic, postfix)

def f_load_saldat_dict(saldict, ds, topic):
    k = f_create_key(ds, topic)
    if k not in saldict:
        print ('Loading saliency maps for ds: {} topic: {}'.format(ds, topic))
        saldict[k] = pickle.load(open(f'{SAL_PATH}/saliency_ds{ds}_topic{topic}', 'rb'), encoding='latin1')
    
def f_load_vector_ds(f_ds, f_topic):
    if f_topic == '5part1':
        f_topic = '5'
    elif f_topic == '6part1':
        f_topic = '6'
    dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(f_ds, f_topic)
    series_ds = headoren.load_series_ds(filename_list, f_parse)
    vector_ds = headoren.headpos_to_headvec(series_ds, f_extract_direction)
    return vector_ds

def f_load_vectords_dict(vector_ds_dict, ds, topic):
    k = f_create_key(ds, topic)
    if k not in vector_ds_dict:
        print ('Loading vector_ds for ds: {} topic: {} '.format(ds, topic))
        vector_ds_dict[f_create_key(ds, topic)] = f_load_vector_ds(ds, topic)