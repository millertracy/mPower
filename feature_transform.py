import numpy as np
import pandas as pd


def filter_mat(X_id, uids):
    """
    Takes in a feature matrix with a uid column as last col
    Returns a filterd feature matrix based on uids

    Example:

    X_id = [[0,0,0,1]
            [1,1,1,2]
            [2,2,2,2]
            [3,3,3,3]]

    uids = [1,3]

    X = [[0,0,0,1]
         [3,3,3,2]]
    """

    Xf = X_id[np.isin(X_id[:,-1], uids)]
    return Xf


def mean_by_uid(X_id, include = False):
    uids = np.unique(X_id[:, -1])
    new_mat = np.zeros((len(uids), X_id.shape[1]))
    count = 0
    for i in range(0, len(uids)):
        count += 1
        if count % 500 == 0:
            print(count)
        ind = X_id[np.where(X_id[:, -1] == uids[i])]
        new_mat[i,:] = ind.mean(axis = 0)

    if not include:
        return new_mat
    else:
        uids = np.array(uids)
        new_mat = np.hstack((new_mat, uids))
        return new_mat



def ufunc_mean(X_id, include = False):
    """
    Takes in an audio feature matrix with an id column as last column and
    Returns collapsed audio feature matrix by taking the mean of each row

    Parameters:
    ----------
    X_id - numpy matrix (each column represents samples across time)
    include - if True include ids as last col

    Output:
    ------
    X - numpy matrix

    Example:
        X_id = np.array([[0,0,0,1],
                         [2,2,2,1],
                         [2,2,2,2],
                         [4,4,4,2]])

        X = [[1,1,1],
             [3,3,3]]

    """

    uids = np.unique(X_id[:, -1])
    _,idx,tags = np.unique(X_id[:,-1], return_index=1, return_inverse=1)
    X = np.add.reduceat(X_id[:,:-1], idx, axis = 0)/np.bincount(tags).reshape(-1,1)

    if not include:
        return X
    else:
        X = np.hstack((X, uids.reshape(-1,1)))
        return X



def ufunc(X_id, stat = 'mean', include = False):
    """
    Takes in an audio feature matrix with an id column as last column and
    Returns collapsed audio feature matrix by taking the mean of each row

    Counts of unique ids must be the same length

    Parameters:
    ----------
    X_id - numpy matrix (each column represents samples across time)
    stat - statistic to apply across each uid
    include - if True include ids as last col

    Output:
    ------
    X - numpy matrix

    Example:
        X_id = np.array([[0,0,0,1],
                         [2,2,2,1],
                         [2,2,2,2],
                         [4,4,4,2]])

        X = [[1,1,1],
             [3,3,3]]

    """
    df = pd.DataFrame(X_id)
    df = df.groupby(df.columns[-1])
    if stat == 'mean':
        return df.mean().as_matrix()
    if stat == 'var':
        return df.var().as_matrix()
    if stat == 'std':
        return df.std().as_matrix()



def invert(X_id, include = False):
    """
    Takes in a matrix with uids as last column
    Returns a pivoted matrix where each row represents responses for 1 uid

    Example:

    X_id = [[0,1]
            [1,1]
            [2,1]
            [1,2]
            [2,2]
            [3,2]]

    inv_mat = [[0,1,2]
               [1,2,3]]
    """

    uids, sample_lens = np.unique(X_id[:, -1], return_counts = True)
    trunc = min(sample_lens)
    shape1 = X_id.shape[1] - 1
    inv_mat = np.zeros((len(uids)*shape1, trunc))
    row = 0
    count = 0
    for i in uids:
        count += 1
        if count % 2000 == 0:
            print(count)
        inv_mat[row:row + shape1, :] = (X_id[np.where(X_id[:,-1] == i)])[0:trunc,:-1].T
        row += shape1

    if not include:
        return inv_mat
    else:
        uids_app = [[i]*shape1 for i in uids]
        uids_app = np.array(uids_app).reshape(-1,1)
        inv_mat = np.hstack((inv_mat, uids_app))
        return inv_mat
