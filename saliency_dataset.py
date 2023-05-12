import pickle
import glob
import os
import imp
import numpy as np
import cv2
from matplotlib import pyplot as plt

#import saliency_ds_lib
import header_saliency_ds as header 
import headoren_sal_corr_helper

class SalnetDatasetGenerator:
    #input file template
    DIRPATH_VIDEO = './data/pano-videos'
    DIRPATH_SALIENCY = '../hmd-observe-video-prediction/data/pano-saliency-merge'
    PATH_SALIENCY_GT       = '../hmd-observe-video-prediction/data/pano-saliency-merge/saliency_ds{}_topic{}'
    PATH_SALIENCY_PRED     = '../hmd-observe-video-prediction/data/pano-saliency-45x80mergepreddirectly-pred152-step002-iter197/saliency_ds{}_topic{}'
    #output

    FILETEMPLATE_DS_DCNN_STEP = '{}/ds_dcnn_step{}'
    FILETEMPLATE_DS_DCNN_TEST = '{}/ds_dcnn_test'
    FILETEMPLATE_DS_DCNN_FULL = '{}/ds_dcnn_full'
    #FILETEMPLATE_SALIENCY = f'{}/saliency_ds{}_topic{}'


    def __init__(self, tdict, sal_dict, fps_dict, MODE):
        #GOAL: prepare dataset to train saliency predictor (resnet model)
        #TODO: create ds_full file 
        #INPUT: pano-saliency folder & pano-vid/frames folder
        #OUTPUT: ds_full file storing ALL (image & fixation input) & saliency ground truth
                #ds_train_step file storing file, with steps
        self.tdict = tdict
        self.saldat_dict = sal_dict
        self.vector_ds_dict = {}
        self.fps_dict = fps_dict
        self.helper_salgt = headoren_sal_corr_helper.HeadorenSalCorrHelper(self.saldat_dict, self.vector_ds_dict, MODE, self.PATH_SALIENCY_GT, self.PATH_SALIENCY_PRED)
        
    
    def get_image(self, ds, topic, t0, resize=True):
        #handle special case when two names the same
        if ds==3 and topic=='diving':
            topic = 'diving2'
        fps = self.fps_dict[ds][topic]
        frameid = int(t0 * fps)
        frame_filepath = f'{self.DIRPATH_VIDEO}/frames/{topic}_{frameid:04d}.jpg'
        image = plt.imread(frame_filepath)
        if resize==True:
            image = cv2.resize(image, (header.TARGET_IMG_W, header.TARGET_IMG_H))
        return image, frame_filepath   
    
    def find_timeindex(self, t0, t_list):
        #TODO: find the nearest index in the given time list
        #INPUT: target time t0, time_list
        #OUTPUT: the index for t_list
        if t0 < t_list[0]: 
            return 0
        if t0 > t_list[-1]:
            return len(t_list) - 1
        for idx, _ in enumerate(t_list[1:]):
            if t_list[idx] > t0 and t_list[idx-1] <= t0:
                return idx
        print (t0, idx, t_list[idx], t_list[idx-1])
        raise #you should not get here
    
    def get_saliencymap(self, saldat, ds, topic, t0, blur=True, resize=True):
        t_list = [item[0] for item in saldat]
        idx = self.find_timeindex(t0, t_list)
        _, _, salmap = saldat[idx]
        if blur==True:
            salmap = cv2.GaussianBlur(salmap,(header.TARGET_GAUSSIAN_BLUR, header.TARGET_GAUSSIAN_BLUR),0)
        if resize==True:
            salmap = cv2.resize(salmap, (header.TARGET_SAL_W, header.TARGET_SAL_H))
        if ds==3 and topic=='panel':
            h, w = salmap.shape
            d_roll = int(h * 0.08)
            salmap = np.roll(salmap, d_roll, axis=0)
        return salmap
    
    def get_sample(self, ds, topic, t0):
            #HOT FIX for framename incompatible
            convert_dict = {'5part1':'5', '6part1':'6'}
            self.helper_salgt.f_load_saldat_dict(ds, topic)
            saldat = self.helper_salgt.saldat_dict[self.helper_salgt.f_create_key(ds, topic)]
            #if topic in convert_dict:
            #    img, img_fp = self.get_image(ds, convert_dict[topic], t0)
            #else:
            img, img_fp = self.get_image(ds, topic, t0)
            smap = self.get_saliencymap(saldat, ds, topic, t0)
            return img, smap
        
    def gen_train_dataset(self, step):
        #TODO1: Create trainset for saliency predictor at step
        sal_ds = []
        bpos, epos, step = 4, 60, step
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                k = self.helper_salgt.f_create_key(ds, topic)
                try:
                    for t0 in np.arange(bpos, epos, step):
                        img, smap = self.get_sample(ds, topic, t0)
                        sal_ds.append((t0, img, smap))
                except Exception as e:
                    print (f"Error at :{ds} - {topic} - {t0} - {e}")
                    continue
        pickle.dump(sal_ds, open(self.FILETEMPLATE_DS_DCNN_STEP.format(self.DIRPATH_SALIENCY, step), 'wb'))
        
    def gen_test_dataset(self):
        #TODO2: Create test set by getting frame at 4.5, 5.5,  ... timestamp
        sal_ds_test = []
        bpos, epos, step = 4.5, 60, 1.0
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                k = self.helper_salgt.f_create_key(ds, topic)
                try:
                    for t0 in np.arange(bpos, epos, step):
                        img, smap = self.get_sample(ds, topic, t0)
                        sal_ds_test.append(((ds, topic, t0), img, smap))
                except Exception as e:
                    print (f"Error at :{ds} - {topic} - {t0} - {e}")
                    continue
        pickle.dump(sal_ds_test, open(self.FILETEMPLATE_DS_DCNN_TEST.format(self.DIRPATH_SALIENCY), 'wb'))
    
    def gen_whole_dataset(self):
        tdict = header.topic_dict
        time_dict = {1:{}, 2:{}, 3:{}}
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                #in_filename = self.FILETEMPLATE_SALIENCY.format(self.DIRPATH_SALIENCY, ds, topic)
                in_filename = self.PATH_SALIENCY_GT.format(ds, topic)
                dat = pickle.load(open(in_filename, 'rb'))
                time_dict[ds][topic] = [item[0] for item in dat]
        salpred_ds =[]
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                k = self.helper_salgt.f_create_key(ds, topic)
                try:
                    for t0 in time_dict[ds][topic]:
                        img, smap = self.get_sample(ds, topic, t0)
                        salpred_ds.append(((ds, topic, t0), img, smap))
                except Exception as e:
                    print (f"Error at :{ds} - {topic} - {t0} - {e}")
                    continue
        pickle.dump(salpred_ds, open(self.FILETEMPLATE_DS_DCNN_FULL.format(self.DIRPATH_SALIENCY), 'wb'))
        
        
class SalnetDatasetGenerator_ExperimentDistance(SalnetDatasetGenerator):
    #input
    PATH_SALIENCY_GT       = '../hmd-observe-video-prediction/data/pano-saliency-merge/saliency_distance_ds{}_topic{}'
    PATH_SALIENCY_PRED     = '../hmd-observe-video-prediction/data/pano-saliency-expdistance/saliency_distance_ds{}_topic{}'    
    DIRPATH_VIDEO = './data/pano-videos'
    DIRPATH_SALIENCY = '../hmd-observe-video-prediction/data/pano-saliency-merge'
    #output
    FILETEMPLATE_DS_DCNN_STEP = '{}/ds_dcnn_expdistance_step{}'
    FILETEMPLATE_DS_DCNN_TEST = '{}/ds_dcnn_expdistance_test'
    FILETEMPLATE_DS_DCNN_FULL = '{}/ds_dcnn_expdistance_full'
    #FILETEMPLATE_SALIENCY = f'{}/saliency_ds{}_topic{}'

class SaliencyDatasetFull(SalnetDatasetGenerator):
    #TODO: this class read frames from both ./data/pano-videos and ./data/pano-videos-background
    #NOTE: THIS CLASS DOES NOT NEED GROUND TRUTH, RETURN NONE
    PATH_SALIENCY_GT         = '../hmd-observe-video-prediction/data/pano-saliency-merge/saliency_distance_ds{}_topic{}'
    PATH_SALIENCY_PRED       = '../hmd-observe-video-prediction/data/pano-saliency-expdistance/saliency_distance_ds{}_topic{}'    
    DIRPATH_VIDEO            = './data/pano-videos'
    DIRPATH_VIDEO_BACKGROUND = './data/pano-videos-background'
    DIRPATH_SALIENCY = '../hmd-observe-video-prediction/data/pano-saliency-merge'
    #output
    FILETEMPLATE_DS_DCNN_STEP = '{}/dsbg_dcnn_expdistance_step{}'
    FILETEMPLATE_DS_DCNN_TEST = '{}/dsbg_dcnn_expdistance_test'
    FILETEMPLATE_DS_DCNN_FULL = '{}/dsbg_dcnn_expdistance_full'
    
    DS_BACKGROUND = 0
    FPS           = 10
    
    def get_topic_background(self):
        fp_list = glob.glob(self.DIRPATH_VIDEO_BACKGROUND + '/frames/*')
        vidid_list = []
        for fp in fp_list:
            filename, ext = os.path.splitext(fp)
            basename = os.path.basename(filename)
            vidid_list.append(basename)
        return {self.DS_BACKGROUND: vidid_list}
    
    def __init__(self, MODE):
        #GOAL: prepare dataset to train saliency predictor (resnet model)
        #TODO: create ds_full file 
        #INPUT: pano-saliency folder & pano-vid/frames folder
        #OUTPUT: ds_full file storing ALL (image & fixation input) & saliency ground truth
                #ds_train_step file storing file, with steps
        self.tdict = header.topic_dict
        self.tdict.update(self.get_topic_background())
        self.saldat_dict = {}
        self.vector_ds_dict = {}
        self.fps_dict = header.fps_dict
        self.fps_dict.update({0:{vid:self.FPS for vid in self.tdict[self.DS_BACKGROUND]}})
        self.helper_salgt = headoren_sal_corr_helper.HeadorenSalCorrHelper(self.saldat_dict, self.vector_ds_dict, MODE, self.PATH_SALIENCY_GT, self.PATH_SALIENCY_PRED)
    
    def get_frame_filepath(self, topic, frameid):
        if topic in header.topic_list:
            return f'{self.DIRPATH_VIDEO}/frames/{topic}_{frameid:04d}.jpg'
        else:
            return f'{self.DIRPATH_VIDEO_BACKGROUND}/frames/{topic}/{frameid:04d}.jpg'
    
    def get_image(self, ds, topic, t0, resize=True):
        #handle special case when two names the same
        if ds==3 and topic=='diving':
            topic = 'diving2'
        fps = self.fps_dict[ds][topic]
        frameid = int(t0 * fps)
        frame_filepath = self.get_frame_filepath(topic, frameid)
        image = plt.imread(frame_filepath)
        if resize==True:
            image = cv2.resize(image, (header.TARGET_IMG_W, header.TARGET_IMG_H))
        return image, frame_filepath   
    
    def get_sample(self, ds, topic, t0):
            img, img_fp = self.get_image(ds, topic, t0)
            if ds in set([1, 2, 3]):
                convert_dict = {'5part1':'5', '6part1':'6'}
                self.helper_salgt.f_load_saldat_dict(ds, topic)
                saldat = self.helper_salgt.saldat_dict[self.helper_salgt.f_create_key(ds, topic)]
                smap = self.get_saliencymap(saldat, ds, topic, t0)
            else:
                smap = None
            return img, smap
        
    def gen_train_dataset(self, step):
        print ("this class don't to gen train dataset, sorry")
        raise
        
    def gen_whole_dataset(self):
        tdict = header.topic_dict
        time_dict = {1:{}, 2:{}, 3:{}, 0:{}}
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                if topic in set([1, 2, 3]):
                    in_filename = self.PATH_SALIENCY_GT.format(ds, topic)
                    dat = pickle.load(open(in_filename, 'rb'))
                    time_dict[ds][topic] = [item[0] for item in dat]
                else:
                    time_dict[ds][topic] = list(np.arange(1, 60, 0.06))
        salpred_ds =[]
        for ds in self.tdict:
            for topic in self.tdict[ds]:
                k = self.helper_salgt.f_create_key(ds, topic)
                try:
                    for t0 in time_dict[ds][topic]:
                        img, smap = self.get_sample(ds, topic, t0)
                        salpred_ds.append(((ds, topic, t0), img, smap))
                except Exception as e:
                    print (f"Error at :{ds} - {topic} - {t0} - {e}")
                    continue
        pickle.dump(salpred_ds, open(self.FILETEMPLATE_DS_DCNN_FULL.format(self.DIRPATH_SALIENCY), 'wb'))
