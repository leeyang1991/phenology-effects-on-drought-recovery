# coding=utf-8
'''
Author: Yang Li
Description: This script contains the key algorithms for the paper of Li et al. (2023)
            "Widespread vegetation phenology effects on drought recovery of Northern Hemisphere ecosystems"
'''
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import stats
from scipy.cluster import hierarchy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import copy
import semopy
from collections import defaultdict


class HANTS:

    def __init__(self):
        '''
        HANTS algorithm for time series smoothing
        '''
        pass

    def makediag3d(self,M):
        b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
        b[:, ::M.shape[1] + 1] = M
        return b.reshape((M.shape[0], M.shape[1], M.shape[1]))

    def get_starter_matrix(self,base_period_len, sample_count, frequencies_considered_count):
        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images
        mat = np.zeros(shape=(nr, sample_count))
        mat[0, :] = 1
        ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
        cs = np.cos(ang)
        sn = np.sin(ang)
        # create some standard sinus and cosinus functions and put in matrix
        i = np.arange(1, frequencies_considered_count + 1)
        ts = np.arange(sample_count)
        for column in range(sample_count):
            index = np.mod(i * ts[column], base_period_len)
            # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
            mat[2 * i - 1, column] = cs.take(index)
            mat[2 * i, column] = sn.take(index)
        return mat

    def hants(self,sample_count, inputs,
              frequencies_considered_count=3,
              outliers_to_reject='Hi',
              low=0., high=255,
              fit_error_tolerance=5.,
              delta=0.1):
        """
        Function to apply the Harmonic analysis of time series applied to arrays
        sample_count    = nr. of images (total number of actual samples of the time series)
        base_period_len    = length of the base period, measured in virtual samples
                (days, dekads, months, etc.)
        frequencies_considered_count    = number of frequencies to be considered above the zero frequency
        inputs     = array of input sample values (e.g. NDVI values)
        ts    = array of size sample_count of time sample indicators
                (indicates virtual sample number relative to the base period);
                numbers in array ts maybe greater than base_period_len
                If no aux file is used (no time samples), we assume ts(i)= i,
                where i=1, ..., sample_count
        outliers_to_reject  = 2-character string indicating rejection of high or low outliers
                select from 'Hi', 'Lo' or 'None'
        low   = valid range minimum
        high  = valid range maximum (values outside the valid range are rejeced
                right away)
        fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
                fit are rejected)
        dod   = degree of overdeterminedness (iteration stops if number of
                points reaches the minimum required for curve fitting, plus
                dod). This is a safety measure
        delta = small positive number (e.g. 0.1) to suppress high amplitudes
        """
        # define some parameters
        base_period_len = sample_count  #

        # check which setting to set for outlier filtering
        if outliers_to_reject == 'Hi':
            sHiLo = -1
        elif outliers_to_reject == 'Lo':
            sHiLo = 1
        else:
            sHiLo = 0

        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images

        # create empty arrays to fill
        outputs = np.zeros(shape=(inputs.shape[0], sample_count))

        mat = self.get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

        # repeat the mat array over the number of arrays in inputs
        # and create arrays with ones with shape inputs where high and low values are set to 0
        mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
        p = np.ones_like(inputs)
        p[(low >= inputs) | (inputs > high)] = 0
        nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

        # prepare for while loop
        ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

        dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
        noutmax = sample_count - nr - dod
        for _ in range(sample_count):
            if ready.all():
                break
            # print '--------*-*-*-*',it.value, '*-*-*-*--------'
            # multiply outliers with timeseries
            za = np.einsum('ijk,ik->ij', mat, p * inputs)

            # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
            diag = self.makediag3d(p)
            A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
            # add delta to suppress high amplitudes but not for [0,0]
            A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
            A[:, 0, 0] = A[:, 0, 0] - delta

            # solve linear matrix equation and define reconstructed timeseries
            zr = np.linalg.solve(A, za)
            outputs = np.einsum('ijk,kj->ki', mat.T, zr)

            # calculate error and sort err by index
            err = p * (sHiLo * (outputs - inputs))
            rankVec = np.argsort(err, axis=1, )

            # select maximum error and compute new ready status
            maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
            ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)

            # if ready is still false
            if not ready.all():
                j = rankVec.take(sample_count - 1, axis=-1)

                p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                    int)  # *check
                nout += 1

        return outputs


class Phenology:

    def __init__(self):
        '''
        # step1: get the 365-day NDVI time series using the function self.hants_smooth
        # step2: get phenology information using the function self.pick_early_peak_late_dormant_period
        '''
        pass

    def hants_smooth(self,NDVI_bi_week):
        '''
        :param NDVI_bi_week: bi-weekly NDVI values
        :return: 365-day NDVI time series
        '''
        NDVI_bi_week = np.array(NDVI_bi_week)
        std = np.nanstd(NDVI_bi_week)
        std = float(std)
        if std == 0:
            return None
        xnew, ynew = self.__interp(NDVI_bi_week)
        ynew = np.array([ynew])
        results = HANTS().hants(sample_count=365, inputs=ynew, low=-10000, high=10000,
                        fit_error_tolerance=std)
        result = results[0]

        return result

    def pick_early_peak_late_dormant_period(self,NDVI_daily,threshold=0.3):
        '''
        :param NDVI_daily: 365-day NDVI time series
        :param threshold: SOS and EOS threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: details of phenology
        '''
        peak = np.argmax(NDVI_daily)
        if peak == 0 or peak == (len(NDVI_daily)-1):
            raise
        try:
            early_start = self.__search_SOS(NDVI_daily, peak, threshold)
            late_end = self.__search_EOS(NDVI_daily, peak, threshold)
        except:
            early_start = np.nan
            late_end = np.nan
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak) # unstable
        # method 2
        early_end, late_start = self.__median_early_late(NDVI_daily,early_start,late_end,peak) # choose the median value before and after the peak

        early_length = early_end - early_start
        mid_length = late_start - early_end
        late_length = late_end - late_start
        dormant_length = 365 - (late_end - early_start)

        result = {
            'early_length':early_length,
            'mid_length':mid_length,
            'late_length':late_length,
            'dormant_length':dormant_length,
            'early_start':early_start,
            'early_start_mon':self.__doy_to_month(early_start),
            'early_end':early_end,
            'early_end_mon':self.__doy_to_month(early_end),
            'peak':peak,
            'peak_mon':self.__doy_to_month(peak),
            'late_start':late_start,
            'late_start_mon':self.__doy_to_month(late_start),
            'late_end':late_end,
            'late_end_mon':self.__doy_to_month(late_end),
            'growing_season':list(range(early_start,late_end)),
            'growing_season_mon':[self.__doy_to_month(i) for i in range(early_start,late_end)],
            'dormant_season':[i for i in range(0,early_start)]+[i for i in range(late_end,365)],
            'dormant_season_mon':[self.__doy_to_month(i) for i in range(0,early_start)]+[self.__doy_to_month(i) for i in range(late_end,365)],
        }
        return result

    def __doy_to_month(self,doy):
        '''
        :param doy: day of year
        :return: month
        '''
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def __interp(self, vals):
        '''
        :param vals: bi-weekly NDVI values
        :return: 365-day NDVI time series with linear interpolation
        '''
        inx = list(range(len(vals)))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)
        return x_new, y_new

    def __search_SOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the Start of Season (SOS)
        '''
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        if left_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            left_min = 2000
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_EOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the End of Season (EOS)
        '''
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        if right_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            right_min = 2000
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold: # stop search when the value is lower than threshold
                break
        return ind

    def __slope_early_late(self,vals,sos,eos,peak):
        slope_left = []
        for i in range(sos,peak):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak,eos):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind

    def __median_early_late(self,vals,sos,eos,peak):
        '''
        :param vals: 365-day NDVI time series
        :param sos: the index of the Start of Season (SOS)
        :param eos: the index of the End of Season (EOS)
        :param peak: the index of the peak index
        :return: the index of the early end and late start
        '''
        median_left = int((peak-sos)/2.)
        median_right = int((eos - peak)/2.)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind


class Single_Drought_Events:

    def __init__(self):
        '''
        step1: pick single drought events using the function self.pick_single_event
        step2: add drought timing to drought events using the function self.add_drought_timing_to_events
        '''
        pass

    def pick_single_event(self,SPEI):
        '''
        :param SPEI: SPEI time series
        :return: a list of single drought events
        '''
        n = 48 # two or more drought events happened in 48 months (4 years) are not considered as single events
        threshold = -2 # SPEI threshold for extreme drought
        events = self.__pick_drought_events(SPEI,threshold)
        if len(events) == 0:
            return

        single_event = []
        for i in range(len(events)):
            if i - 1 < 0:  # The very first event
                if events[i][0] - n < 0 or events[i][-1] + n >= len(SPEI):  # Avoid the very first drought event which is too close to the boundary
                    continue
                if len(events) == 1: # Only one event
                    single_event.append(events[i])
                elif events[i][-1] + n <= events[i + 1][0]:
                    single_event.append(events[i])
                continue

            if i + 1 >= len(events): # The very last event
                if events[i][0] - events[i - 1][-1] >= n and events[i][-1] + n <= len(SPEI): # Avoid the very last drought event which is too close to the boundary
                    single_event.append(events[i])
                break

            # The middle events
            if events[i][0] - events[i - 1][-1] >= n and events[i][-1] + n <= events[i + 1][0]: # The remaining events
                single_event.append(events[i])
        return single_event

    def add_drought_timing_to_events(self,phenology_list:list,SPEI,drought_events:list):
        '''
        :param phenology_list: is the output of Phenology class, monthly, multiyear
        :param SPEI: SPEI time series
        :param drought_events: a list of picked single drought events
        :return: a list of drought events with timing information
        '''
        events_with_timing = []
        for event in drought_events:
            min_indx = self.__pick_min_indx(SPEI,event) # pick the index of minimum SPEI value in the drought event
            year_index = min_indx // 12
            mon = min_indx % 12 + 1 # index to month
            early_start, early_end, late_start, late_end = phenology_list[year_index]
            early_gs = list(range(early_start, early_end + 1))  # early growing season
            peak_gs = list(range(early_end + 1, late_start))  # peak growing season
            late_gs = list(range(late_start, late_end + 1))  # late growing season
            if mon in early_gs:
                timing = 'early'
            elif mon in peak_gs:
                timing = 'mid'
            elif mon in late_gs:
                timing = 'late'
            else:
                timing = 'dormant'
            events_with_timing.append([event,timing])
        return events_with_timing

    def __pick_drought_events(self, SPEI,threshold):
        '''
        :param SPEI: SPEI time series
        :param threshold: SPEI threshold for extreme drought
        :return: a list of drought events
        '''
        drought_month = []
        for i, val in enumerate(SPEI):
            if val < threshold:
                drought_month.append(i)
            else:
                drought_month.append(-999999)
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []
        return events

    def __pick_min_indx(self, array, indexs):
        '''
        :param array: a time series
        :param indexs: a list of indexs
        :return: the index of the minimum value in the time series
        '''
        min_index = 99999
        min_val = 99999
        for i in indexs:
            val = array[i]
            if val < min_val:
                min_val = val
                min_index = i
        return min_index


class Recovery_Time:

    def __init__(self):
        '''
        calculate the recovery time of drought events using the function self.calculate_recovery_time
        '''
        pass

    def calculate_recovery_time(self, SPEI, NDVI, single_events_with_timing, phenology_list, landcover_type):
        '''
        :param SPEI: SPEI time series
        :param NDVI: NDVI time series
        :param single_events_with_timing: a list of drought events with timing, output of Single_Drought_Events class
        :param phenology_list: is the output of Phenology class, monthly, multiyear
        :param landcover_type: landcover type, Deciduous, Evergreen, Shrubland, Grassland from GLC2000
        :return: a list of drought events with recovery informations
        '''
        recovery_time_result = []
        for timing,event_index in single_events_with_timing:
            event_start_index = self.__pick_min_indx(SPEI,event_index) # pick the index of minimum SPEI value in the drought event as the start of the drought event
            year_indx = event_start_index // 12 # the year of the drought event
            early_start, early_end, late_start, late_end = phenology_list[year_indx] # phenology of the drought year
            growing_season = list(range(early_start, late_end + 1)) # growing season of the drought year

            recovery_range, lag, recovery_start_gs, recovery_start, recovery_mode = \
                self.__information_on_the_drought_process(NDVI, event_start_index, growing_season, landcover_type) # vegegation recovery details on the drought process
            if recovery_range == None:
                continue
            recovery_range = np.array(recovery_range)
            drought_event_date_range = np.array(event_index)
            recovery_time = len(recovery_range)
            recovery_time_result.append({
                'recovery_time': recovery_time,
                'recovery_date_range': recovery_range,
                'drought_event_date_range': drought_event_date_range,
                'timing': timing,
                'lag': lag,
                'recovery_start_gs': recovery_start_gs,
                'recovery_mode': recovery_mode,
            })
        return recovery_time_result

    def __search_for_recovery_time(self,recovery_start,event_start_index,growing_season,NDVI,ndvi_threshold,landcover_type):
        '''
        :param recovery_start: the start of the recovery process
        :param event_start_index: the start of the drought event
        :param growing_season: the growing season of the drought year
        :param NDVI: NDVI time series
        :param ndvi_threshold: the threshold of normal NDVI
        :param landcover_type: landcover type, Deciduous, Evergreen, Shrubland, Grassland from GLC2000
        :return: drought recovery date range and recovery mode
        '''
        success = False # initialize the success flag
        recovery_mode = 'Rsgs' # initialize the recovery mode
        greater_than_3_consecutive_month_flag = 0 # initialize the flag for consecutive months
        recovery_range = []

        for i in range(recovery_start, event_start_index + 36):
            mon = i % 12 + 1
            if not mon in growing_season:
                recovery_mode = 'Rmgs'
                continue
            if event_start_index + 36 >= len(NDVI):
                break
            ndvi_i = NDVI[i]
            mon = i % 12 + 1

            if landcover_type == 'evergreen': # for evergreen, dormant season is not considered
                if ndvi_i > ndvi_threshold:
                    greater_than_3_consecutive_month_flag+=1
                    if greater_than_3_consecutive_month_flag > 2: # NDVI needs to exceed the threshold for more than 3 consecutive months
                        success = True
                        break
                else:
                    greater_flag = 0
            else: # for other landcover types, dormant season is considered
                if ndvi_i > ndvi_threshold and mon in growing_season:
                    greater_than_3_consecutive_month_flag+=1
                    if greater_than_3_consecutive_month_flag > 2: # NDVI needs to exceed the threshold for more than 3 consecutive months
                        success = True
                        break
                else:
                    greater_than_3_consecutive_month_flag = 0
            recovery_range.append(i)

        if success == False:
            return None, None
        return recovery_range, recovery_mode

    def __information_on_the_drought_process(self, NDVI, event_start_index, growing_season, lc_type):
        '''
        :param NDVI: NDVI time series
        :param event_start_index: the month of minimum SPEI value in the drought event as the start of the drought event
        :param growing_season: growing season of the drought year
        :param lc_type: landcover type, Deciduous, Evergreen, Shrubland, Grassland from GLC2000
        :return: recovery range, lag, which growing season recovery started, recovery start, recovery mode
        '''
        ndvi_threshold = .0 # NDVI anomaly threshold

        picked_ndvi_vals = []
        picked_ndvi_vals_i = []

        picked_ndvi_index = []
        picked_ndvi_index_i = []
        for i in range(36): # Search for information only 3 years after the drought happened
            if (event_start_index + i) >= len(NDVI):  # if the drought event happened in the last year of the time series, then stop searching
                break
            search_ = event_start_index + i
            search_mon = search_ % 12 + 1
            # only search in growing season
            if not search_mon in growing_season:
                if len(picked_ndvi_vals_i) != 0:
                    picked_ndvi_vals.append(picked_ndvi_vals_i)
                    picked_ndvi_index.append(picked_ndvi_index_i)
                picked_ndvi_vals_i = []
                picked_ndvi_index_i = []
            else:
                ndvi_i = NDVI[search_]
                picked_ndvi_vals_i.append(ndvi_i)
                picked_ndvi_index_i.append(search_)
        if len(picked_ndvi_vals) == 0: # if no growing season after the drought event, then return None
            return None, None, None, None, None

        first_gs_min_ndvi = min(picked_ndvi_vals[0])
        if len(picked_ndvi_vals) == 1:
            second_gs_min_ndvi = 999999
        else:
            second_gs_min_ndvi = min(picked_ndvi_vals[1])
        if first_gs_min_ndvi < ndvi_threshold: # drought recovery happend in the first growing season after the drought event
            min_indx = self.__pick_min_indx(picked_ndvi_vals[0], range(len(picked_ndvi_vals[0])))
            recovery_start = picked_ndvi_index[0][min_indx]
            lag = recovery_start - event_start_index
            recovery_range, recovery_mode = \
                self.__search_for_recovery_time(recovery_start,event_start_index,growing_season,ndvi,ndvi_threshold,lc_type)
            recovery_start_gs = 'first'

        elif second_gs_min_ndvi < ndvi_threshold: # drought recovery happend in the second growing season after the drought event
            recovery_start = picked_ndvi_index[1][0]
            recovery_range, recovery_mode = \
                self.__search_for_recovery_time(recovery_start, event_start_index, growing_season, ndvi, ndvi_threshold,lc_type)
            lag = recovery_start - event_start_index
            recovery_start_gs = 'second'
        else:
            recovery_start = None
            lag = None
            recovery_range = None
            recovery_mode = None
            recovery_start_gs = None

        return recovery_range, lag, recovery_start_gs, recovery_start, recovery_mode

    def __pick_vals_from_1darray(self, array, index):
        '''
        :param array: 1d array
        :param index: index of the array
        :return: values of the array at the index
        '''
        picked_vals = []
        for i in index:
            picked_vals.append(array[i])
        picked_vals = np.array(picked_vals)
        return picked_vals

    def __pick_min_indx(self, array, indexs):
        '''
        :param array: a time series
        :param indexs: a list of indexs
        :return: the index of the minimum value in the time series
        '''
        min_index = 99999
        min_val = 99999
        for i in indexs:
            val = array[i]
            if val < min_val:
                min_val = val
                min_index = i
        return min_index


class Factor_Importance:

    def __init__(self):
        '''
        This class is used to calculate the importance of each factor to the drought recovery
        '''
        pass

    def remove_variables_collinearity(self,dataframe, xvar_list, t=0.0):
        '''
        :param dataframe: a dataframe
        :param xvar_list: a list of x variables
        :param t: threshold
        :return: a list of x variables that are not correlated with each other
        Reference:
        https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        '''
        df = dataframe.dropna() # drop rows with missing values
        X = df[xvar_list] # x variables
        corr = np.array(X.corr())
        corr_linkage = hierarchy.ward(corr) # hierarchical clustering
        cluster_ids = hierarchy.fcluster(corr_linkage, t=t, criterion='distance') # cluster the variables
        cluster_id_to_feature_ids = defaultdict(list) # create a dictionary
        for idx, cluster_id in enumerate(cluster_ids): # add the variables to the dictionary
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features_indx = [v[0] for v in cluster_id_to_feature_ids.values()] # select the first variable in each cluster
        selected_features = []
        for i in selected_features_indx:
            selected_features.append(xvar_list[i])
        return selected_features

    def random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        variable_list = self.remove_variables_collinearity(X, variable_list) # remove variables that are correlated with each other
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=7) # calculate the importance of each variable using permutation importance
        importances = result.importances_mean # get the importance of each variable
        importances_dic = dict(zip(variable_list, importances)) # put the importance of each variable into a dictionary
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        y_pred = clf.predict(X_test) # predict the y variable using the testing data
        r_model = stats.pearsonr(Y_test, y_pred)[0] # calculate the correlation between the predicted y variable and the actual y variable
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred) # calculate the mean squared error
        score = clf.score(X_test, Y_test) # calculate the R^2
        return clf, importances_dic, mse, r_model, score, Y_test, y_pred


class Partial_Dependence_Plots:

    def __init__(self):
        '''
        This class is used to get the partial dependence plots
        Reference:
        https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
        '''
        pass

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=4) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})


class SEM:

    def __init__(self):
        '''
        This class is used to calculate the structural equation model
        '''
        pass

    def model_description(self):
        desc = '''
        # regressions
        current_sos_std_anomaly ~ dormant_SWE_terra + dormant_TMP + PRE_MR_drought_start_to_eos + TMP_MR_drought_start_to_eos + VPD_MR_drought_start_to_eos
        recovery_time ~ PRE_second_year_spring  + TMP_second_year_spring + dormant_SWE_terra + dormant_TMP + PRE_MR_drought_start_to_eos + TMP_MR_drought_start_to_eos + VPD_MR_drought_start_to_eos + current_sos_std_anomaly
        # residual correlations
        current_sos_std_anomaly~~current_sos_std_anomaly
        current_sos_std_anomaly~~dormant_SWE_terra
        current_sos_std_anomaly~~dormant_TMP
        current_sos_std_anomaly~~PRE_MR_drought_start_to_eos
        current_sos_std_anomaly~~TMP_MR_drought_start_to_eos
        current_sos_std_anomaly~~VPD_MR_drought_start_to_eos

        recovery_time~~recovery_time
        recovery_time~~PRE_second_year_spring
        recovery_time~~TMP_second_year_spring
        recovery_time~~dormant_SWE_terra
        recovery_time~~dormant_TMP
        recovery_time~~PRE_MR_drought_start_to_eos
        recovery_time~~TMP_MR_drought_start_to_eos
        recovery_time~~VPD_MR_drought_start_to_eos
        recovery_time~~current_sos_std_anomaly
        '''
        return desc

    def build_model(self,df):
        '''
        :param df: a dataframe
        :return: a SEM model and output a report
        '''
        desc = self.model_description()
        df = df[df['recovery_mode'] == 'Rmgs']
        df = df.dropna(subset=['lc'])
        mod = semopy.Model(desc)
        res = mod.fit(df)
        semopy.report(mod, "SEM_report")
