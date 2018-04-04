import os
import re
import json
import arff
import numpy as np
import pandas as pd
from collections import defaultdict

py_path = '/Users/mill/my_galv/capstone/mPower/Sage-Parkinsons/Voice/Parkinsons_mPower_Voice_Features/'
gmaps_path = '/Users/mill/my_galv/capstone/mPower/gmaps/'
hcs = {}


def extract_pyfeatures(group, path = py_path):
    """
    Takes in a pandas df and path
    Return 3 voice feature matrices and metadata df

    Parameters:
    ----------
    group - pandas dataframe
    path - absolute path to voice features data

    Output:
    ------
    MFCC - numpy matrix of MFCC features + uid for all individuals
           in the group
    chroma_vector - numpy matrix of chroma vector features + uid
                    for all individuals in the group
    other_feats - numpy matrix of rest of voice features + uid
                  for all individuals in the group
    feat_info - pandas dataframe contains metadata for the individuals
                and the voice task

    """

    #initialize feature vars (larger than need to be) and last col = uid
    #faster than stacking
    MFCC = np.zeros((59000*410, 14))
    chroma_vector = np.zeros((59000*410, 13))
    other_feats = np.zeros((59000*410, 10))

    #create dict to capture feature metadata
    d = defaultdict(list)

    #create dictionary to get diagnoses
    hcs = set(group['healthCode'].values)
    diag_dict = {hc: diag for hc,diag in zip(group['healthCode'].values, group['diag'].values)}

    #get length of directory
    len_dir = len(os.listdir(py_path))

    #initialize count,row
    count = 0
    row = 0


    for file in os.listdir(path):
        count += 1
        if count%5000 == 0:
            percent = (count/len_dir)*100
            print('{}% completed'.format(round(percent,2)))

        if (file.endswith(".json")):
            if (file.endswith("NLX-1.json")):
                continue

            file_id = file.split(".")[0]
            with open(path + file) as js:
                data = json.load(js)
                if data['healthcode'] not in hcs:
                    continue
                else:

                    #get the diagnosis, create uid per individual
                    diag = diag_dict[data['healthcode']]
                    uid = count

                    #append ind features + uid to feature vars
                    mfcc = np.array(data['features']['audio']['MFCC']).T
                    cv = np.array(data['features']['audio']['chroma_vector']).T

                    dim1 = mfcc.shape[0]
                    uid_arr = np.array([uid] * dim1).reshape(dim1,-1)
                    mfcc = np.concatenate((mfcc, uid_arr), axis = 1)
                    cv = np.concatenate((cv, uid_arr), axis = 1)

                    ofs = np.array(data['features']['audio']['ZCR']).reshape(-1,1)
                    for x in list(data['features']['audio'].keys())[1:]:
                        if x in ['MFCC', 'chroma_vector']:
                            continue
                        feat = np.array(data['features']['audio'][x]).reshape(-1,1)
                        ofs = np.concatenate((ofs, feat), axis = 1)
                    ofs = np.concatenate((ofs, uid_arr), axis = 1)

                    MFCC[row : row + mfcc.shape[0], :] = mfcc
                    chroma_vector[row : row + mfcc.shape[0], :] = cv
                    other_feats[row : row + mfcc.shape[0], :] = ofs

                    #increase row var
                    row += mfcc.shape[0]

                    #append metadata to d
                    d['fid'].append(file_id)
                    d['uid'].append(uid)
                    d['healthCode'].append(data['healthcode'])
                    d['phoneinfo'].append(data['phoneinfo'])
                    d['appversion'].append(data['appversion'])
                    d['medtimepoint'].append(data['medtimepoint'])
                    d['sample_len'].append(mfcc.shape[0])
                    d['diagnosis'].append(diag_dict[data['healthcode']])


    #remove extra all 0 rows
    MFCC = MFCC[~np.all(MFCC==0, axis = 1)]
    chroma_vector = chroma_vector[~np.all(chroma_vector==0, axis = 1)]
    other_feats = other_feats[~np.all(other_feats==0, axis = 1)]

    #create a feature info df that corresponds to feature feature matrix
    feat_info = pd.DataFrame(d)


    return MFCC, chroma_vector, other_feats, feat_info


def extract_gmaps(path):
    """
    Takes in an absolute path
    Returns a default dictionary of GeMaps voice features
    in key, value pairs
    """

    count = 0
    len_dir = len(os.listdir(path))

    d = defaultdict(list)
    for file in os.listdir(path):
        if file.endswith('.arff'):
            count += 1
            if count%5000 == 0:
                percent = (count/len_dir)*100
                print('{}% completed'.format(round(percent,2)))
            re_obj = re.search('(\d+[^m4a])', file)
            file_id = re_obj.group()[:-1]
            d['fid'].append(file_id)
            data = arff.load(open(path + file), 'rb')
            for i in range(len(data['attributes'])):
                d[data['attributes'][i][0]].append(data['data'][0][i])
    return d
