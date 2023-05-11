import numpy as np
import pickle
import cv2

import header_headoren_salcorr as header
import saldat_head_orientation
import head_orientation_lib

#THIS FILE CONTAINS CLASSES RESPONSIBLE FOR MANAGING VECTOR_DS, MATCHING VECTOR_DS WITH SALIENCY FOR CORRELATION PURPOSE
#ALSO, IT HAS HEADOREN AS A PROPERTY, THUS, IT IS USE IT TO HANDLE SALIENCY MAPS, IF POSSIBLE

#
    
##############    

class HeadorenSalCorrHelper:
    
    #PATH_SALIENCY_GT       = './data/pano-saliency-merge/saliency_ds{}_topic{}'
    #PATH_SALIENCY_PRED     = './data/pano-saliency-45x80merge-pred101-step002-iter418/saliency_ds{}_topic{}'
    #PATH_SALIENCY_PRED     = './data/pano-saliency-45x80mergepreddirectly-pred152-step002-iter197/saliency_ds{}_topic{}'
    
    PATH_SAL_ACTIVITY_GT   = './data/sal_activity_gt_dict'
    PATH_SAL_ACTIVITY_PRED = './data/sal_activity_pred_dict'
    
    PATH_MEAN_SALMAP_GT    = './data/mean_salmap_gt'
    PATH_MEAN_SALMAP_PRED  = './data/mean_salmap_pred'
    

    
    def __init__(self, saldat_dict, vector_ds_dict, saliency_mode, path_saliency_gt, path_saliency_pred):
        self.name = "HeadorenSalCorrHelper"
        
        self.saliency_mode = saliency_mode
        if saliency_mode not in {'groundtruth', 'predicted'}:
            print ('ERROR, saliency_mode must be either "groundtruth" or "predicted"')
            raise
            
        self.PATH_SALIENCY_GT = path_saliency_gt
        self.PATH_SALIENCY_PRED = path_saliency_pred
        #must specify mode (groundtruth or predicted)
        #self._saliency_ds_gt_filepath = './data/pano-saliency/saliency_ds{}_topic{}'
        #self._saliency_ds_pred_filepath = './data/pano-saliency-pred/saliency_ds{}_topic{}'
        
        #keep headoren variable internally
        print ('LOG: initilize headoren')
        self.headoren = saldat_head_orientation.HeadOrientation(header.dirpath1, header.dirpath2, header.dirpath3, header.ext1, header.ext2, header.ext3)
        
        ##this one specifically just for step 2 of our project   
        print ('LOG: loading topic dict')
        self.tdict   = header.topic_dict
        
        print ('LOG: initilize saldat dict & vector ds dict')
        self.saldat_dict    = saldat_dict
        self.vector_ds_dict = vector_ds_dict
        
        if saliency_mode == 'groundtruth':
            self.sactivity_path = self.PATH_SAL_ACTIVITY_GT
            self.saldat_path = self.PATH_SALIENCY_GT
            self.mean_salmap_path = self.PATH_MEAN_SALMAP_GT
        elif saliency_mode == 'predicted':
            self.sactivity_path = self.PATH_SAL_ACTIVITY_PRED
            self.saldat_path = self.PATH_SALIENCY_PRED
            self.mean_salmap_path = self.PATH_MEAN_SALMAP_PRED
        
        print ('LOG: initilize mean salmap')
        try: 
            self.mean_salmap = self.load_mean_salmap()
        except:
            self.mean_salmap = self.create_mean_salmap()
        #self.mean_salmap    = (self.mean_salmap - self.mean_salmap.mean())/self.mean_salmap.std()
        
        print ('LOG: initilize sactivity dict')
        try:
            self.sactivity_dict = pickle.load(open(self.sactivity_path, 'rb'))
        except:
            self.create_sactivity_dict()
            
    #GENERIC INITILIZATION
    def f_create_key(self, ds, topic):
        return '{}_{}'.format(ds, topic)
    
    def f_parse_key(self, k):
        return k.split('_')
    
    def create_exp_dictkey(self, f_ds, s_ds, f_topic, s_topic):
        return '{}_{}_{}_{}'.format(f_ds, s_ds, f_topic, s_topic)

    def parse_exp_dictkey(self, k):
        f_ds, s_ds, f_topic, s_topic = k.split('_')
        f_ds = int(f_ds)
        s_ds = int(s_ds)
        return f_ds, s_ds, f_topic, s_topic
    
    def f_load_saldat_dict(self, ds, topic):
        k = self.f_create_key(ds, topic)
        if k not in self.saldat_dict:
            filepath = self.saldat_path.format(ds, topic)
            print (f'Loading saliency maps for ds: {ds} topic: {topic} at {filepath}')
            self.saldat_dict[k] = pickle.load(open(filepath, 'rb'), encoding='latin1')
            
    def convert_to_nonstrict_monotonic(self, vec):
        idx_list = []
        for idx, _ in enumerate(vec[:-1]):
            if vec[idx+1][0] - vec[idx][0] < 0:
                vec[idx+1][0] = vec[idx][0]
        return vec

    def filter_vector(self, vec):
        #remove large changes: very big number before the starting of the array
        idx_list = []
        min_pos = np.argmin([item[0] for item in vec])
        min_val = np.min([item[0] for item in vec])
        for idx, _ in enumerate(vec[:-1]):
            if vec[idx][0] > min_val and idx < min_pos:
                idx_list.append(idx)
        idx_list = set(idx_list)
        vec = [item for idx, item in enumerate(vec) if idx not in idx_list]

        #remove small changes
        vec = self.convert_to_nonstrict_monotonic(vec)
        #now find all non monotonic position
        idx_list = []
        for idx, _ in enumerate(vec[:-1]):
            if vec[idx+1][0] == vec[idx][0]:
                idx_list.append(idx+1)

        if len(idx_list) != len(set(idx_list)):
            raise
        idx_list = set(idx_list)
        vec = [item for idx, item in enumerate(vec) if idx not in idx_list]
        return vec
    
    def filter_vector_ds(self, vector_ds):
        for idx, _ in enumerate(vector_ds):
            vec = vector_ds[idx]
            vec = self.filter_vector(vec)
            vector_ds[idx] = vec
        return vector_ds

    def f_load_vector_ds(self, f_ds, f_topic):
#         #only this code should have this if since it must conform the public format
#         #need to remove in fugure
#         if f_topic == '5part1':
#             f_topic = '5'
#         elif f_topic == '6part1':
#             f_topic = '6'
#         elif f_ds == 3 and f_topic == 'diving2':
#             f_topic = 'diving'
#         dirpath, filename_list, f_parse, f_extract_direction = \
#                                     self.headoren.load_filename_list(f_ds, f_topic)
#         series_ds = self.headoren.load_series_ds(filename_list, f_parse)
#         vector_ds = self.headoren.headpos_to_headvec(series_ds, f_extract_direction)
        vector_ds = self.headoren.load_vector_ds(dataset, topic)
        vector_ds = self.filter_vector_ds(vector_ds)
        return vector_ds

    def f_load_vector_ds_dict(self, ds, topic):
        k = self.f_create_key(ds, topic)
        if k not in self.vector_ds_dict:
            print ('Loading vector_ds for ds: {} topic: {} '.format(ds, topic))
            self.vector_ds_dict[self.f_create_key(ds, topic)] = self.f_load_vector_ds(ds, topic)

    #MEAN SALMAP (FOR SPATIAL CORR)
    def f_normalize_salmap(self, salmap):
        if salmap.sum() != 0:
            return (salmap - salmap.mean())/(salmap.std())
        else:
            return salmap
    
    def create_mean_salmap(self):
        result = np.zeros((head_orientation_lib.H, head_orientation_lib.W))
        for ds in header.topic_dict:
            for topic in header.topic_dict[ds]:
                try:
                    filepath = self.saldat_path.format(ds, topic)
                    salmap_list = pickle.load(open(filepath, 'rb'))
                    for _, _, salmap in salmap_list:
                        result += salmap
                except:
                    continue
        result = self.f_normalize_salmap(result)
        pickle.dump(result, open(self.mean_salmap_path, 'wb'))
        return result
    
    def load_mean_salmap(self):
        return pickle.load(open(self.mean_salmap_path, 'rb'))

    #SACTIVITY DICT (FOR TEMPORAL CORR) 
    def extract_salmap_window(self, saldat, t0, window_size):
        #TODO: extract saliency map from given window time (t0, t0 + widow_size). Why?
        timestamp_list, vlist, salmap_list = zip(*saldat)
        idx_begin, idx_end = 1, 1
        for i in range(len(timestamp_list)):#salmap_list
            if timestamp_list[i] < t0:
                idx_begin += 1
            if timestamp_list[i] < t0 + window_size:
                idx_end += 1
        salmap_window = salmap_list[idx_begin:idx_end]            
        return timestamp_list[idx_begin:idx_end], salmap_window
    
    def create_sactivity_dict(self, s_T0=0, N=60):
        self.sactivity_dict = {}
        topic_dict = self.tdict
        for ds in topic_dict:
            for topic in topic_dict[ds]:
                self.f_load_saldat_dict(ds, topic)
                sal_dat = self.saldat_dict[self.f_create_key(ds, topic)]
                s_timestamp, s_window0 = self.extract_salmap_window(sal_dat, s_T0, N)
                self.get_saliency_activity(self.sactivity_dict, ds, topic, s_timestamp, s_window0, N)
        pickle.dump(self.sactivity_dict, open(self.sactivity_path, 'wb'))
        
    def get_saliency_activity(self, dsal_dict, s_ds, s_topic, s_timestamp, s_window0, N):
        k = self.f_create_key(s_ds, s_topic)
        if k not in dsal_dict:
            idx_list, d_list = self.create_saliency_activity(s_window0)
            dat = np.array([idx_list, [s_timestamp[idx] for idx in idx_list], d_list])
            #extract first N seconds of data only
            dat = dat.T[dat[1] <= N].T
            #normalize the data
            dat[2] = (dat[2] - dat[2].mean())/dat[2].std()
            dsal_dict[k] = dat
        return dsal_dict
    
    def create_saliency_activity(self, s_window0, offset=20):
        #return activity happend BEFORE the given idx, specified by offset
        d_list = []
        idx_list = []
        for idx, _ in enumerate(s_window0[offset:]):
            smap_a = s_window0[idx]
            smap_b = s_window0[idx-offset]
            d = ((smap_a - smap_b)**2).sum()
            #d = ((s_window0[idx] - s_window0[idx-offset])**2).sum()
            d_list.append(d)    
            idx_list.append(idx)
        d_list = np.array(d_list)
        d_list = (d_list - d_list.mean())/np.std(d_list)
        return idx_list, d_list#(d_list-d_list.mean())/np.std(d_list)
    
    #TEMPORAL CORRELATION
    def find_fixation_temporal(self, vec, dataset):
        #TODO: call cutoff_vec_acc to cut the saccade, 
        #       then show the time of fixation right after saccade is cutoff
        result = []
        #cut_vec = self.headoren.cutoff_vel_acc_compliment([vec], dataset, thres_list=(15, 30))[0]
        cut_vec = self.headoren.cutoff_vel_acc_compliment([vec], dataset, thres_list=(25, 50))[0]
        t_list = [item[0] for item in cut_vec if item[0] > 5.]
        for idx, _ in enumerate(t_list[:-1]):
            result.append((idx+1, t_list[idx+1]))
        return result
    
    def match_index(self, timelista, timelistb):
        #TODO: for each time in timelista, find a corresponding index in timelistb
        #INPUT: timelista: time points, where condition happends, 
        #        timelistb: full time series data
        #OUTPUT: all index in timelistb match timelista
        if timelistb[0] > timelista[0]: 
            #potential mismatch happend
            raise
        if timelistb[0] > timelista[-1]:
            #never a match will happend
            raise
        result = []
        b_idx = 0
        for idx,_ in enumerate(timelista):
            while b_idx < len(timelistb) and timelistb[b_idx] < timelista[idx]:
                b_idx += 1
            #prevent adding overflow index to result, stop when detecting one
            if b_idx >= len(timelistb):
                break
            result.append(b_idx)
        return result

    def extract_salmovement(self, cut_timelist, s_timestamp, d_list):
        #TODO: go over cut_timelist, for a given time in cut_timelist, extract the corresponding d(s_window)
        #stidx_list return position in s_timestamp that match cut_timelist, get those saliency!
        result = []
        stidx_list = self.match_index(cut_timelist, s_timestamp)
        for sidx in stidx_list:
            result.append([sidx, s_timestamp[sidx], d_list[sidx]])
        return result
    
    def corr_salmovement_item(self, cut_timelist, s_ds, s_topic):
        sact_idx_list, salact_time_list, salact_list = self.sactivity_dict[self.f_create_key(s_ds, s_topic)]
        tmp = self.extract_salmovement(cut_timelist, salact_time_list, salact_list)
        if len(tmp) == 0:
            return -1
        else:
            return np.mean([item[2] for item in tmp ])
    
    def video_headm_timecorr(self, f_ds, f_topic, s_ds, s_topic, uid, f_T0 = 1, s_T0 = 1, N=300):
        self.f_load_vector_ds_dict(f_ds, f_topic)
        vector_ds = self.vector_ds_dict[self.f_create_key(f_ds, f_topic)]
        self.f_load_saldat_dict(s_ds, s_topic)
        sal_dat = self.saldat_dict[self.f_create_key(s_ds, s_topic)]
        s_timestamp, s_window0 = self.extract_salmap_window(sal_dat, s_T0, N)

        cut_idxlist, cut_timelist = list(zip(*self.find_fixation_temporal(vector_ds[uid], f_ds)))
        sact_idx_list, sact_time_list, sact_list = self.sactivity_dict[self.f_create_key(s_ds, s_topic)]
        corr = self.corr_salmovement_item(cut_timelist, s_ds, s_topic)
        return corr
    
    #SPATIAL CORRELATION
    
    def extract_fixation_window(self, vector_ds, dataset, uid, t0, window_size):
        timestamp_list, _, _, _ = zip(*vector_ds[uid])
        idx_begin, idx_end = 0, 0
        for i in range(len(vector_ds[uid])):
            if timestamp_list[i] < t0:
                idx_begin += 1
            if timestamp_list[i] < t0 + window_size:
                idx_end += 1
        vector_window = vector_ds[uid][idx_begin:idx_end]
        fixation_window = [self.headoren.create_fixation_map([item], dataset) for item in vector_window]
        return timestamp_list[idx_begin:idx_end], fixation_window
    
    def find_fixation_spatial(self, vec, dataset, delta=0.17):
        #TODO: call cutoff_vec_acc to cut the saccade, 
        #       then show the time of fixation right after saccade is cutoff
        result = []
        cut_vec = self.headoren.cutoff_vel_acc([vec], dataset, thres_list=(27, 40))[0]
        t_list = [item[0] for item in cut_vec if item[0] > 5.]
        for idx, _ in enumerate(t_list[:-1]):
            result.append((idx, t_list[idx]))
        return result
    
    def filter_array(self, ft, v_begin, v_end):
        #TODO: extract the ft array from values v_begin and v_end
        #INPUT: array with float values, threshold values v_begin, v_end
        #OUTPUT: new filtered array
        result = []
        idx_list = []
        idx = 0
        while ft[idx] < v_begin:
            idx += 1
        while idx < len(ft) and ft[idx] <= v_end :
            result.append(ft[idx])
            idx_list.append(idx)
            idx += 1
        return idx_list, result
    
    def extract_fixation(self, cutvec_timelist, f_time, s_time, delta=0.06):
        #TODO: match f_time with s_time given cutvec_timelist
        result = []
        c_idx0, f_idx, s_idx = 0, 0, 0;
        #first, must expect that s_idx & f_idx <= c_idx, or else all will fail
        #increase c_idx until cutvec_timelist[c_idx] > f_time[f_dix] and cutvec_timelist[c_idx] > s_time[s_dix]
        while cutvec_timelist[c_idx0] < f_time[f_idx] or cutvec_timelist[c_idx0] < s_time[s_idx]:
            c_idx0 += 1
        #assuming now that c_idx is in safe indexes
        for c_idx in range(c_idx0, len(cutvec_timelist)):
            try:
                while f_idx < len(f_time) and np.abs(cutvec_timelist[c_idx] - f_time[f_idx]) > delta:
                    f_idx += 1
                while s_idx < len(s_time) and np.abs(cutvec_timelist[c_idx] - s_time[s_idx]) > delta:
                    s_idx += 1
                if f_idx == len(f_time) or s_idx == len(s_time):
                    break
                result.append([c_idx, f_idx, s_idx, cutvec_timelist[c_idx], f_time[f_idx], s_time[s_idx]])
            except Exception as ex:
                print ('Error at: ', c_idx, f_idx, s_idx, len(cutvec_timelist), len(f_time), len(s_time), ex)
                raise
                break
        return result
    
    def elimbias(self, salmap, factor=0.85):
        if self.saliency_mode == 'groundtruth':
            salmap = self.f_normalize_salmap(cv2.GaussianBlur(salmap, (11, 11), 0)) 
            salmap0 = self.f_normalize_salmap(cv2.GaussianBlur(self.mean_salmap, (11, 11), 0))
        elif self.saliency_mode == 'predicted':
            salmap = self.f_normalize_salmap(salmap)
            salmap0 = self.f_normalize_salmap(self.mean_salmap)
        else:
            raise
        result = salmap - (salmap0 * factor)
        result[result<0] = 0
           
        return result
    
    def salcorr(self, vizdat, f_window0, s_window0):
        #TODO: calculate the correlation through times between head orientation & saliency maps
        #output: list of correlation coef for each point specified by c_idx
        result = []
        for c_idx, f_idx, s_idx, c_t, f_t, s_t in vizdat:    
            dst = self.elimbias(s_window0[s_idx], factor=0.9)
            result.append(dst[f_window0[f_idx]>0].sum())
        result = np.array(result)
        result = result[~np.isnan(result)]
        return result
    
    def video_headm_spatialcorr(self, f_ds, f_topic, s_ds, s_topic, uid, f_T0=5, s_T0=5, N=300):
        #TODO: calculate the correlation between head orientation & saliency maps
        #INPUT: ds, topic, & data dict
        #OUTPUT: the array of correlation coefficient
        self.f_load_vector_ds_dict(f_ds, f_topic)
        vector_ds = self.vector_ds_dict[self.f_create_key(f_ds, f_topic)]
        self.f_load_saldat_dict(s_ds, s_topic)
        sal_dat = self.saldat_dict[self.f_create_key(s_ds, s_topic)]

        f_timestamp, f_window0 = self.extract_fixation_window(vector_ds, f_ds, uid, f_T0, N)
        s_timestamp, s_window0 = self.extract_salmap_window(sal_dat, s_T0, N)
        cut_idxlist, cut_timelist = list(zip(*self.find_fixation_spatial(vector_ds[uid], f_ds)))

        #GOINT BACK TO USING FIXATION AFTER SACCADE TO EXATRACT SPATIAL CORRELATION
        #_, cut_timelist1 = self.filter_array(cut_timelist, f_T0, 999)
        viz_dat = self.extract_fixation(cut_timelist, f_timestamp, s_timestamp, delta=0.08)

        return self.salcorr(viz_dat, f_window0, s_window0), viz_dat
    
    
class HeadorenSalCorrStep2Helper(HeadorenSalCorrHelper):
    #NOTE: vector_ds & fixation maps collected are from 
    #SOURCE_VECTOR_DS_GROUNDTRUTH = 'groundtruth'
    #SOURCE_VECTOR_DS_PREDICTED = 'predicted'
    
    def __init__ (self, saldat_dict, vector_ds_dict, vector_ds_source_mode, saliency_mode, path_saliency_gt, path_saliency_pred):
        HeadorenSalCorrHelper.__init__(self, saldat_dict, vector_ds_dict, saliency_mode)
        self.name = "HeadorenSalCorrStep2Helper"
        self.headoren = saldat_head_orientation.HeadOrientationStep2(vector_ds_source_mode)
        self.tdict  = header.topic_step2_dict
        
    def f_load_vector_ds(self, f_ds, f_topic):
#         if f_topic == '5part1':
#             f_topic = '5'
#         elif f_topic == '6part1':
#             f_topic = '6'
        
        vector_ds = self.headoren.load_vector_ds(f_ds, f_topic)
        vector_ds = self.filter_vector_ds(vector_ds)
        return vector_ds            
                