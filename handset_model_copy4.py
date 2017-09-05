from collections import Counter
# import sys
import os.path
import h5py
import argparse
import hashlib
# import MySQLdb
import pickle
# import _mysql_exceptions
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, Input, Flatten, Activation, concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras.layers.advanced_activations import PReLU
import keras.backend as K

# Categorical: CU_MAP_SEGMENT, CU_ADSL_OK_RESULT, CU_ADSL_MAX_DL_SPEED, CU_VDSL_OK_RESULT, CU_LINE_STATUS: Nulls to nans -> 0
# Categorical: HH_MOB_COVERAGE_IN", "HH_MOB_COVERAGE_OU": UKJENT to nans -> 0
# Numeric: Null to 0
# OBS!: MPP_DEVICE_MODELNAME/Id... and CU_U_MAIN_DEV_MODELNAME/ID, "CU_U_MAIN_DEV_OS_TYPE",
#                       "CU_U_MAIN_DEV_PRODUCERNAME", "CU_U_MAIN_DEV_CATEGORY", "CU_U_MAIN_DEV_TYPE",
#                       "CU_U_MAIN_DEV_HD_VOICE", "CU_U_MAIN_DEV_TOUCH_SCREEN", "CU_U_MAIN_DEV_LTE" should have same embedding

SEED = 7
N_SAMPLES = 583291 * 0.8

CATEGORICAL_COLS = ["CU_GENDER", "CU_EMAIL_IND", "CU_SMS_IND", 
                     "CU_MAP_SEGMENT", "CLM_LIVSFASE_SEGMENT", "CU_FYLKE", 
                     "CU_HOUSE_TYPE", "HH_MOB_COVERAGE_IN", "HH_MOB_COVERAGE_OUT", 
                     "CU_ADSL_OK_RESULT", "CU_VDSL_OK_RESULT", "CU_LINE_STATUS",
                     "CU_U_MAIN_BRAND", "CU_U_MAIN_SUBS_TYPE", "CU_U_MAIN_DEV_MODELNAME", 
                     "CU_U_MAIN_DEV_OS_TYPE", "CU_U_MAIN_DEV_PRODUCERNAME", "CU_U_MAIN_DEV_CATEGORY", 
                     "CU_U_MAIN_DEV_TYPE", "CU_U_MAIN_DEV_HD_VOICE", 
                     "CU_U_MAIN_DEV_TOUCH_SCREEN", "CU_U_MAIN_DEV_LTE", "MPP_BRAND", 
                     "MPP_SUBS_START_REASON", "MPP_CHANGETYPE", "MPP_PRODUCT_NAME", 
                     "MPP_CHURN_SEGMENT_ADJ", "MPP_DEVICE_MODELNAME", "MPP_DEVICE_OS_TYPE", 
                     "MPP_DEVICE_PRODUCERNAME", "MPP_DEVICE_CATEGORY", "MPP_DEVICE_TYPE", 
                     "MPP_DEVICE_HD_VOICE", "MPP_DEVICE_TOUCH_SCREEN", "MPP_DEVICE_LTE", 
                     "MPP_BINDING_PRODUCT_DESC", "MPP_BINDING_TYPE", "MPP_DEALER_NAME", 
                     "MPP_SALES_CH", "MPP_DEALER_CHAIN", "MPP_IN_PORT_SP_NAME"]

COLS = ['TARGET_S_TO_S_APPLE'] + CATEGORICAL_COLS

COLS_TERM = ["TARGET_S_TO_S_APPLE", "CU_AGE", "CU_GENDER", "CU_U_MAIN_DEV_MODELNAME", 
             "CU_U_MAIN_DEV_PRODUCERNAME", "MPP_DEVICE_LTE", 
             "MPP_BINDING_NO_DAYS_ACTIVE", "MPP_DEVICE_OS_TYPE", 
             "MPP_DEVICE_PRODUCERNAME", "MPP_KR_MMS_DOM_LAST2", "MPP_MB_LAST2", "MPP_MB_LAST3", 
             "MPP_GROSS_PERIODIC_FEE_FULL",
             "MPP_NET_OTHER_FEE", "MPP_NET_REVENUE", "MPP_NO_SMS_DOM_LAST3", "MPP_NO_SMS_SUM_3MO", 
             "MPP_NO_TERMINAL_BUNDELS", "MPP_NO_VOICE_DOM_LAST3"]



_DROP_COLS = ["ID", "CU_U_MAIN_SUBSCRIPTION_KEY", "CU_U_MAIN_DEV_MODEL_ID", "MPP_DEVICE_MODEL_ID", "X_LIVSFASE",
             "MPP_NET_DISCOUNT_OTHER_FEE"]
DROP_COLS = []

LABEL_COL = "TARGET_S_TO_S_APPLE"

_CATEGORICAL_COLS = ['CU_MAP_SEGMENT', 'CLM_LIVSFASE_SEGMENT', 'CU_U_MAIN_DEV_OS_TYPE',
                    'CU_U_MAIN_DEV_PRODUCERNAME', 'MPP_DEVICE_OS_TYPE', 'MPP_DEVICE_PRODUCERNAME',
                    'CU_U_MAIN_DEV_MODELNAME', 'MPP_DEVICE_MODELNAME']


_BINARY_COLS = ["CU_EMAIL_ACCEPT_FLAG", "CU_SMS_AKSEPT_FLAG", "CU_RES_TELENOR_DM", "CU_RES_TELENOR_TM",
               "CU_RES_BRSUND_TM", "CU_RES_BRSUND_DM", "MPP_NEW_USER_IND", "MPP_NEWSALE_IND", "MPP_SHOPPER_FLG",
               "MPP_FRIFAM_FLG", "MPP_FORSIKRING_FLG", "MPP_MIN_SKY_FLG", "MPP_BINDING_ACTIVE_FLG"]
BINARY_COLS = []

_X_CATEGORICAL_COLS = ["X_ENSLIG_PAR", "X_POSTREKLAMESEGMENTER_LAST1", "X_BOLIGTYPE_ENKEL", "X_BOLIGTYPE_ENKEL_HIST",
                      "X_MERKE_NYESTE_BIL", "X_BILSEGMENT_NYESTE_BIL"]
X_CATEGORICAL_COLS = []
_X_BINARY_COLS = ["X_HUSEIER", "X_HYTTEEIER"]
X_BINARY_COLS =[]

_CAT_EMB_DIM = {'X_POSTREKLAMESEGMENTER_LAST1': 10, 'X_BILSEGMENT_NYESTE_BIL': 5, 'CU_U_MAIN_DEV_MODELNAME': 76,
               'CU_GENDER': 2, 'X_ENSLIG_PAR': 2, 'HH_MOB_COVERAGE_IN': 2,
               'CU_U_MAIN_DEV_PRODUCERNAME': 16, 'CLM_LIVSFASE_SEGMENT': 6, 'CU_EMAIL_IND': 2,
               'CU_U_MAIN_DEV_CATEGORY': 5, 'MPP_DEVICE_MODELNAME': 77,
               'CU_U_MAIN_DEV_OS_TYPE': 5, 'CU_SMS_IND': 2, 'HH_MOB_COVERAGE_OUT': 2, 'MPP_BRAND': 1,
               'CU_U_MAIN_SUBS_TYPE': 2, 'CU_MAP_SEGMENT': 6, 'CU_LINE_STATUS': 3, 'CU_U_MAIN_DEV_HD_VOICE': 2,
               'CU_FYLKE': 19, 'X_BOLIGTYPE_ENKEL': 6, 'CU_U_MAIN_DEV_TOUCH_SCREEN': 2, 'CU_ADSL_OK_RESULT': 4,
               'CU_HOUSE_TYPE': 6, 'X_MERKE_NYESTE_BIL': 32, 'MPP_PRODUCT_NAME': 10, 'MPP_BINDING_PRODUCT_DESC': 32,
               'MPP_DEVICE_PRODUCERNAME': 16, 'MPP_DEVICE_CATEGORY': 5, 'MPP_DEALER_NAME': 94,
               'CU_U_MAIN_DEV_TYPE': 9, 'MPP_SUBS_START_REASON': 3, 'MPP_DEVICE_LTE': 2, 'MPP_DEALER_CHAIN': 16,
               'CU_VDSL_OK_RESULT': 4, 'MPP_CHURN_SEGMENT_ADJ': 10, 'CU_U_MAIN_DEV_LTE': 2, 'MPP_DEVICE_OS_TYPE': 7,
               'MPP_SALES_CH': 12, 'MPP_BINDING_TYPE': 5, 'CU_U_MAIN_BRAND': 2, 'MPP_DEVICE_TYPE': 9,
               'MPP_CHANGETYPE': 3, 'X_BOLIGTYPE_ENKEL_HIST': 9, 'MPP_DEVICE_HD_VOICE': 2, 'MPP_IN_PORT_SP_NAME': 9,
               'MPP_DEVICE_TOUCH_SCREEN': 1}

_CAT_EMB_DIM = {"CU_GENDER": 2, "CU_U_MAIN_DEV_MODELNAME":76, "CU_U_MAIN_DEV_PRODUCERNAME":16, "MPP_DEVICE_LTE":2, 
             "MPP_DEVICE_OS_TYPE": 7, "MPP_DEVICE_PRODUCERNAME":16}

_CAT_EMB_DIM = {'CU_MAP_SEGMENT': 2, 'CLM_LIVSFASE_SEGMENT': 6, 'CU_U_MAIN_DEV_OS_TYPE': 5,
               'CU_U_MAIN_DEV_PRODUCERNAME': 16, 'MPP_DEVICE_OS_TYPE': 7, 'MPP_DEVICE_PRODUCERNAME': 16,
               'CU_U_MAIN_DEV_MODELNAME': 76, 'MPP_DEVICE_MODELNAME': 77}
CAT_EMB_DIM = dict()

# Original dimension minus one
# CAT_EMB_DIM = {'X_POSTREKLAMESEGMENTER_LAST1': 10, 'X_BILSEGMENT_NYESTE_BIL': 11, 'CU_U_MAIN_DEV_MODELNAME': 1535,
#                'CU_GENDER': 2, 'X_ENSLIG_PAR': 2, 'HH_MOB_COVERAGE_IN': 2,
#                'CU_U_MAIN_DEV_PRODUCERNAME': 179, 'CLM_LIVSFASE_SEGMENT': 6, 'CU_EMAIL_IND': 2,
#                'CU_U_MAIN_DEV_CATEGORY': 10, 'MPP_DEVICE_MODELNAME': 842,
#                'CU_U_MAIN_DEV_OS_TYPE': 25, 'CU_SMS_IND': 2, 'HH_MOB_COVERAGE_OUT': 2, 'MPP_BRAND': 1,
#                'CU_U_MAIN_SUBS_TYPE': 2, 'CU_MAP_SEGMENT': 6, 'CU_LINE_STATUS': 3, 'CU_U_MAIN_DEV_HD_VOICE': 2,
#                'CU_FYLKE': 19, 'X_BOLIGTYPE_ENKEL': 9, 'CU_U_MAIN_DEV_TOUCH_SCREEN': 2, 'CU_ADSL_OK_RESULT': 4,
#                'CU_HOUSE_TYPE': 6, 'X_MERKE_NYESTE_BIL': 106, 'MPP_PRODUCT_NAME': 62, 'MPP_BINDING_PRODUCT_DESC': 142,
#                'MPP_DEVICE_PRODUCERNAME': 57, 'MPP_DEVICE_CATEGORY': 8, 'MPP_DEALER_NAME': 3286,
#                'CU_U_MAIN_DEV_TYPE': 9, 'MPP_SUBS_START_REASON': 3, 'MPP_DEVICE_LTE': 2, 'MPP_DEALER_CHAIN': 57,
#                'CU_VDSL_OK_RESULT': 4, 'MPP_CHURN_SEGMENT_ADJ': 10, 'CU_U_MAIN_DEV_LTE': 2, 'MPP_DEVICE_OS_TYPE': 20,
#                'MPP_SALES_CH': 16, 'MPP_BINDING_TYPE': 5, 'CU_U_MAIN_BRAND': 2, 'MPP_DEVICE_TYPE': 8,
#                'MPP_CHANGETYPE': 3, 'X_BOLIGTYPE_ENKEL_HIST': 9, 'MPP_DEVICE_HD_VOICE': 2, 'MPP_IN_PORT_SP_NAME': 9,
#                'MPP_DEVICE_TOUCH_SCREEN': 1}

# List of lists (empty by default), where each sub-list contains a number of categorical variables (identified by their
# 0-based index in the CATEGORICAL_COLS list) that take same set of values
_SHARED_EMBEDDING = [[7, 8], [9, 10], [12, 22], [14, 27], [15, 28], [16, 29], [17, 30], [18, 31], [19, 32], [20, 33],
                    [21, 34]]
_SHARED_EMBEDDING = [[2,4], [3,5]]
SHARED_EMBEDDING = []

X_SHARED_EMBEDDING = []

# QUERY_TRAIN = 'SELECT * from crm.S_TO_S_APPLE_TRAIN'
# QUERY_VAL = 'SELECT * from crm.S_TO_S_APPLE_VALID'


# def get_db_connection():
#     return MySQLdb.connect(db='crm', host="10.0.0.236", port=3306, user="root")

def load_data():
    # QUERY = 'SELECT * from crm.S_TO_S_APPLE_TRAIN'
    # cnx = get_db_connection()
    # df = pd.read_sql_query(QUERY, con=cnx)
    # cnx.close()
    
    df = pd.read_csv("handset_data_train_wo_X.csv", usecols=COLS)
    
    return df


# Input: List of categorical variables
# Output: a dictionary where the keys are the names of the categorical variables. The value associated with each key
#         is a list of the possible values the categorical variable can take
def get_distinct_values(df, cat_vars, x_vars=False):
    for var in CATEGORICAL_COLS:
            df.loc[df.loc[:,var].isnull(), var] = "UNKNOWN"
    df.loc[df.MPP_DEVICE_OS_TYPE.str.contains("MICROSOFT"),"MPP_DEVICE_OS_TYPE"] = "MICROSOFT"
    df.loc[df.MPP_DEVICE_OS_TYPE.str.contains("SYMBIAN"),"MPP_DEVICE_OS_TYPE"] = "SYMBIAN"
    df.loc[df.MPP_DEVICE_OS_TYPE.str.contains("BLACKBERRY"),"MPP_DEVICE_OS_TYPE"] = "BLACKBERRY"
    df.loc[df.MPP_DEVICE_OS_TYPE.isin(['LINUX MAEMO', 'ASHA', 'PROPRIETARY OS', 'BADA']),"MPP_DEVICE_OS_TYPE"] = "OTHER"
    df.loc[:, "COUNT"] = df.groupby(["CU_U_MAIN_DEV_PRODUCERNAME"]).CU_U_MAIN_DEV_PRODUCERNAME.transform('count')
    df.loc[df.COUNT<200,"CU_U_MAIN_DEV_PRODUCERNAME"] = "OTHER"
    df.loc[:, "COUNT"] = df.groupby("MPP_DEVICE_PRODUCERNAME").MPP_DEVICE_PRODUCERNAME.transform('count')
    df.loc[df.COUNT<200,"MPP_DEVICE_PRODUCERNAME"] = "OTHER"
    df.loc[df.CU_U_MAIN_DEV_PRODUCERNAME=="OTHER","CU_U_MAIN_DEV_MODELNAME"] = "OTHER"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="IPHONE 5 (A1429)", "CU_U_MAIN_DEV_MODELNAME"]="IPHONE 5"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="IPHONE 5S (A1457)", "CU_U_MAIN_DEV_MODELNAME"]="IPHONE 5S"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="IPHONE 6S (A1688 / A1691 / A1700 / A1633)", "CU_U_MAIN_DEV_MODELNAME"]="IPHONE 6S"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="IPHONE 5C (A1532 / A1456)", "CU_U_MAIN_DEV_MODELNAME"]="IPHONE 5C"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="GT-I9506 (GALAXY S4 LTE-A)", "CU_U_MAIN_DEV_MODELNAME"]="GT-I9505 (GALAXY S4 LTE)"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="SM-G920F (SM-G920I GALAXY S6)", "CU_U_MAIN_DEV_MODELNAME"]="SM-G920F ( GALAXY S6)"
    df.loc[df.CU_U_MAIN_DEV_MODELNAME=="SM-G925F (SM-G925I GALAXY S6 EDGE)", "CU_U_MAIN_DEV_MODELNAME"]="SM-G925F( GALAXY S6 EDGE)"
    df.loc[:, "COUNT"] = df.groupby(["CU_U_MAIN_DEV_PRODUCERNAME","CU_U_MAIN_DEV_MODELNAME"]).CU_U_MAIN_DEV_MODELNAME.transform('count')
    brands = df.CU_U_MAIN_DEV_PRODUCERNAME.unique()
    for brand in brands:
        df.loc[(df.COUNT<1000)&(df.CU_U_MAIN_DEV_PRODUCERNAME==brand), "CU_U_MAIN_DEV_MODELNAME"] = brand + "_other"

    df.loc[:, 'MPP_DEALER_NAME'] = df.MPP_DEALER_NAME.str.lower()
    df.loc[df.MPP_DEALER_NAME.str.contains("telenor"),"MPP_DEALER_NAME"] = "TELENOR"
    df.loc[df.MPP_DEALER_NAME.str.contains("telekiosk"),"MPP_DEALER_NAME"] = "TELENOR"
    df.loc[df.MPP_DEALER_NAME.str.contains("telehus"),"MPP_DEALER_NAME"] = "TELENOR"
    df.loc[df.MPP_DEALER_NAME.str.contains("telering"),"MPP_DEALER_NAME"] = "TELERING"
    df.loc[df.MPP_DEALER_NAME.str.contains("telebutikken"),"MPP_DEALER_NAME"] = "TELEBUTIKKEN"
    df.loc[df.MPP_DEALER_NAME.str.contains("elkjøp"),"MPP_DEALER_NAME"] = "ELKJØP"
    df.loc[df.MPP_DEALER_NAME.str.contains("spaceworld"),"MPP_DEALER_NAME"] = "SPACEWORLD"
    df.loc[df.MPP_DEALER_NAME.str.contains("expert"),"MPP_DEALER_NAME"] = "EXPERT"
    df.loc[df.MPP_DEALER_NAME.str.contains("mobildata"),"MPP_DEALER_NAME"] = "MOBILDATA"
    df.loc[df.MPP_DEALER_NAME.str.contains("lefdal"),"MPP_DEALER_NAME"] = "LEFDAL"
    df.loc[df.MPP_DEALER_NAME.str.contains("nordialog"),"MPP_DEALER_NAME"] = "NORDIALOG"
    df.loc[df.MPP_DEALER_NAME.str.contains("coop"),"MPP_DEALER_NAME"] = "COOP"
    df.loc[df.MPP_DEALER_NAME.str.contains("elektrosenter"),"MPP_DEALER_NAME"] = "ELEKTROSENTER"
    df.loc[df.MPP_DEALER_NAME.str.contains("elprice"),"MPP_DEALER_NAME"] = "ELPRICE"
    df.loc[:, "COUNT"] = df.groupby(["MPP_DEALER_NAME"]).MPP_DEALER_NAME.transform('count')
    df.loc[df.COUNT<200,"MPP_DEALER_NAME"] = "OTHER"
    # # mpp dealer name now has 95 unique values (instead of ~3000!)
    
    df.loc[df.MPP_DEVICE_MODELNAME=="IPHONE 5 (A1429)", "MPP_DEVICE_MODELNAME"]="IPHONE 5"
    df.loc[df.MPP_DEVICE_MODELNAME=="IPHONE 5S (A1457)", "MPP_DEVICE_MODELNAME"]="IPHONE 5S"
    df.loc[df.MPP_DEVICE_MODELNAME=="IPHONE 6S (A1688 / A1691 / A1700 / A1633)", "MPP_DEVICE_MODELNAME"]="IPHONE 6S"
    df.loc[df.MPP_DEVICE_MODELNAME=="IPHONE 5C (A1532 / A1456)", "MPP_DEVICE_MODELNAME"]="IPHONE 5C"
    df.loc[df.MPP_DEVICE_MODELNAME=="GT-I9506 (GALAXY S4 LTE-A)", "MPP_DEVICE_MODELNAME"]="GT-I9505 (GALAXY S4 LTE)"
    df.loc[df.MPP_DEVICE_MODELNAME=="SM-G920F (SM-G920I GALAXY S6)", "MPP_DEVICE_MODELNAME"]="SM-G920F ( GALAXY S6)"
    df.loc[df.MPP_DEVICE_MODELNAME=="SM-G925F (SM-G925I GALAXY S6 EDGE)", "MPP_DEVICE_MODELNAME"]="SM-G925F( GALAXY S6 EDGE)"
    df.loc[:, "COUNT"] = df.groupby(["MPP_DEVICE_PRODUCERNAME","MPP_DEVICE_MODELNAME"]).MPP_DEVICE_MODELNAME.transform('count')
    brands = df.MPP_DEVICE_PRODUCERNAME.unique()
    for brand in brands:
        df.loc[(df.COUNT<1000)&(df.MPP_DEVICE_PRODUCERNAME==brand), "MPP_DEVICE_MODELNAME"] = brand + "_other"
    # mpp device modelname now has 78 unique values
    
    df.drop('COUNT', axis=1, inplace=True)
    
    fname = "cat_levels_{}_x.pickle".format("w" if x_vars else "wo")
    if os.path.isfile(fname):
        print("Loading dict")
        cat_vars_dict = pickle.load(open(fname, "rb"))
    else:
        cat_vars_dict = {}
        # df = df.astype(str)
        # print("3")
        #df.fillna("missing", inplace=True)
        for var in cat_vars:
            df2 = pd.DataFrame(df.loc[:,var])
            cat_vars_dict[var] = list(df2[var].str.lower().unique())  # values are lowercase!

    return cat_vars_dict

def _integer_encode(df, cat_levels):
    df_cat = pd.DataFrame(df[list(cat_levels.keys())].copy())
    count = 0
    for key, values in cat_levels.items():
        count += 1
        print(count)
        if df_cat[key].dtype == 'O':
            # df_cat[key].fillna("missing", inplace=True)
            df_cat.loc[:, key] = df_cat.loc[:, key].astype(str).str.lower()
        # else:
            # df_cat[key].fillna("missing", inplace=True)
        for i in range(len(values)):
            value = values[i]
            df_cat.loc[df_cat[key] == value, key] = i
    return df_cat


# Encodes categorical variables using one-hot encoding. New columns are added to the input dataframe, one for each of
# the possible values that each of the categorical variables can take. Old columns are removed.
# Input:
# - df: Pandas dataframe containing the categorical variables to be encoded as columns
# - cat_levels: Dictionary with the different values each categorical variable can take
# - binary_enc: if True, the negative cases of the generated dummy variables will be represented as 0
#      (i.e., 0/1 values), otherwise as -1 (i.e., -1/1 values)
#
# Output: A sparse dataframe containing the encoded categorical variables.
#

# THIS TAKES A REALLY LONG TIME!
# reduce vars before using
def _one_hot_encode(df, cat_levels, binary_enc=False):
    print("starting one hot encode")
    df_sp = pd.DataFrame()
    neg = 0 if binary_enc else -1
    for key, values in cat_levels.items():
        print(key)
        for value in values:
            # compare lowercase
            df_sp[key + "_" + str(value)] = (df[key].astype('str').str.lower()).apply(lambda x: 1 if x == str(value) else neg)
            # df.drop([key], axis=1, inplace=True)

    return df_sp  #.to_sparse(fill_value=neg)


# It applies the feature hashing trick to encode categorical variables as fixed-sized numerical vectors.
# The values of categorical variables are considered as sequences of chars, so the hashing is applied to each one of
# the characters.
# (https://www.quora.com/Can-you-explain-feature-hashing-in-an-easily-understandable-way)
# Input:
# - df: Pandas dataframe containing the categorical variables to be encoded as columns
# - vars: a list with the names of categorical variables
# - dim: If an integer, the default dimension for the output vectors. If a dictionary, the dimensions to use for each
#       categorical variable
#
#  Output: A sparse dataframe, with dim*len(vars) columns, containing the encoded categorical variables.
#
def _feture_hashing(df, vars, dim, hash_fnc='md5'):
    hasher = hashlib.new(hash_fnc)
    df_sp = pd.DataFrame()
    for var in vars:
        _dim = dim if type(dim) is int else dim[var]
        cols = [var + "_" + str(i) for i in range(_dim)]

        def xform(x):
            tmp = [0 for _ in range(_dim)]
            for c in str(x):
                c = c.encode('utf-8')
                hasher.update(c)
                tmp[int(hasher.hexdigest(), 16) % _dim] += 1
            return pd.Series(tmp, index=cols)
        df_sp[cols] = df[var].apply(xform)
        # df.drop(var, axis=1, inplace=True)
    return df_sp.to_sparse(fill_value=0)


def _standardize(df, mean=None, std=None):
    if mean is None:
        _mean = []
        _std = []
        for col in df.columns:
            _mean.append(df[col].mean())
            _std.append(df[col].std())
            df[col] = (1.0 * (df[col] - _mean[-1])) / _std[-1]
        mean = pd.DataFrame(columns=df.columns)
        mean.loc[0] = _mean
        std = pd.DataFrame(columns=df.columns)
        std.loc[0] = _std
    else:
        for col in df.columns:
            df[col] = (1.0 * (df[col] - mean.loc[0, col])) / std.loc[0, col]

    return mean, std


def _log_transform(df):
    for col in df.columns:
        min = df[col].min()
        if min < 0:
            df[col] = df[col] + abs(min)  # avoid negative values
        df[col] = df[col] + 1  # avoid zeros
        df[col] = df[col].apply(np.log)


# Input:
# - df: Dataframe containing the data to be whitened
# - cold: List with the names of the dataframe columns that hold the data to be whitened
# - whiten: If True, whitening is performed (only on non-categorical and non-binary variables)
# - reduce_dim: {0, 1, 0<x<1} If 0, no dimensionality reduction is done. If 1, Thomas P. Minka's method
#               ("Automatic Choice of Dimensionality for PCA". NIPS 2000) is used to determine the number of dimensions
#               to keep. If 0 < reduce_dim < 1, enough number of dimensions will be kept to keep "reduce_dim" percentage
#               of variance explained.
# - pca: None, or an already fitted PCA model to be applied to test data
#
# Output: A new dataframe with the whitened data (inlcuiding also any data that has not been whitened)
def _pca_whitening(df, cols, whiten=True, reduce_dim=0, pca=None):
    if reduce_dim == 1: reduce_dim = 'mle'
    if reduce_dim == 0: reduce_dim = None
    if pca is None:
        pca = PCA(whiten=whiten, n_components=reduce_dim, svd_solver='full')
        data_new = pca.fit_transform(df[cols].values)
    else:
        data_new = pca.transform(df[cols].values)

    df.drop(cols, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if reduce_dim is not None:
        cols = ["PCA_" + str(i) for i in range(data_new.shape[1])]
    return pd.concat([df, pd.DataFrame(data_new, columns=cols)], axis=1), pca


# Input:
# - cat_encoding: Specifies the method to use to encode categorical variables, either 'hashing', 'one-hot' or 'integer'
#                 integer-based encoding is needed when using Keras embedding layers
# - cat_emb_dim: Needed if hashinh categorical variables. It is a dictionary with the number of dimensions to be used
#               when hash each of the categorical variables. Alternatively, an integer specifying a default hashing
#               dimension
# - standardize: If True, data is normalized to have zero mean and unit variance
# - pca_whiten: If True, data is whitened using PCA whitening (vaue of "normalize" is ignored)
# - pca_reduce: {0, 1, 0<x<1} If 0, no dimensionality reduction is done. If 1, Thomas P. Minka's method
#               ("Automatic Choice of Dimensionality for PCA". NIPS 2000) is used to determine the number of dimensions
#               to keep. If 0 < pca_reduce < 1, enough number of dimensions will be kept to keep "pca_reduce" percentage
#               of variance explained.
#               OBS!!! The MLE method crashes!
# - log_transform: If True, apply a natural logarithm transformation
# - binary_enc: if True, the negative cases of binary variables will be represented as 0
#      (i.e., 0/1 values), otherwise as -1 (i.e., -1/1 values)
#
# Output:
# - df:
# - df_cat:
# - cat_levels: Dictionary with the different values each categorical variable can take
#
# OBS! As this blog post (http://blog.explainmydata.com/2012/07/should-you-apply-pca-to-your-data.html)
# shows, there is no universal recipe for preprocessing. What works for one dataset, might not work for another one.
def preprocess_data(df, cat_levels, data_to_process=['num', 'cat'], cat_encoding='hashing', cat_emb_dim=32, standardize=True,
                    pca_whiten=False, pca_reduce=None, log_transform=False, binary_enc=False, include_x_vars=False,
                    mean=None, std=None, pca=None):
    # labels = np.copy(df[LABEL_COL].values)

    # Drop non-needed variables
    df.drop(DROP_COLS + [LABEL_COL], axis=1, inplace=True)

    if include_x_vars:
        cat_vars = CATEGORICAL_COLS + X_CATEGORICAL_COLS
    else:
        cat_vars = CATEGORICAL_COLS
        # df.drop(X_CATEGORICAL_COLS, axis=1, inplace=True)


    df_cat = None
    if 'cat' in data_to_process:
        # Encode categorical variables.
        print("Encoding categorical variables...")

        if cat_encoding == 'hashing':
            df_cat = _feture_hashing(df, cat_vars, cat_emb_dim, hash_fnc='md5')
        elif cat_encoding == 'one-hot':
            df_cat = _one_hot_encode(df, cat_levels, binary_enc=binary_enc)
        else:
            df_cat = _integer_encode(df, cat_levels)  # doesn't work with unhashable type dict_keys

        df_cat.reset_index(drop=True, inplace=True)

    df.drop(cat_vars, axis=1, inplace=True)

#     if 'num' in data_to_process:
#         # Convert binary variables to [-1,1] range if needed
#         if include_x_vars:
#             bin_vars = BINARY_COLS + X_BINARY_COLS
#         else:
#             bin_vars = BINARY_COLS
#             # df.drop(X_BINARY_COLS, axis=1, inplace=True)

#         if not binary_enc:
#             print("Enconding binary variables...")
#             for var in bin_vars:
#                 df[var] = df[var].apply(lambda x: -1 if x == 0 else x)

#         # Replace NaNs in numeric and binary columns with zeros
#         df.fillna(0, inplace=True)

#         for col in df.columns:
#             if df[col].dtype == 'O':
#                 df[col] = df[col].astype('float32')

#         # Log-transform
#         if log_transform:
#             print("Applying log transformation...")
#             _log_transform(df)

#         # Normalize/Standardize the data
#         if standardize:
#             print("Normalizing and standardizing...")
#             mean, std = _standardize(df, mean, std)

#         # Whiten (only non-categorical and non-binary variables)
#         if pca_whiten:
#             print("PCA whitening...")
#             num_vars = [var for var in df.columns if var not in bin_vars]
#             if not standardize:
#                 if mean is None:
#                     _mean = []
#                     for col in num_vars:
#                         _mean.append(df[col].mean())
#                         df[col] = (df[col] - _mean[-1])
#                     mean = pd.DataFrame(columns=num_vars)
#                     mean.loc[0] = _mean
#                 else:
#                     for col in num_vars:
#                         df[col] = (df[col] - mean.loc[0, col])

#             df, pca = _pca_whitening(df, num_vars, whiten=pca_whiten, reduce_dim=pca_reduce, pca=pca)


#         df.reset_index(drop=True, inplace=True)

    return df, df_cat, mean, std, pca


# Input:
# - dim_num_var: number of numerical and binary variables
# - dim_cat_var: number of categorical variables
# - cat_encoding: Specifies the method used to encode categorical variables, either 'hashing', 'one-hot' or 'integer'
#                 integer-based encoding is needed when using Keras embedding layers
# - cat_emb_dim: An integer representing the dimension to be used for all embedding vectors or a dictionary with
#                the number of dimensions to be used to embed each of the categorical variables
# - cat_levels: Dictionary with the different values each categorical variable can take
#
def create_model(num_vars, cat_vars, cat_encoding='hashing', cat_emb_dim=32, cat_levels=None,
                 include_x_vars=False, activation='prelu'):
    dim_num_var = len(num_vars)
    dim_cat_var = len(cat_vars)
    # Define input tensors ...
    # ... for binary and numeric variables
    num_bin_input = Input(shape=(dim_num_var,), dtype='float32', name='num_bin_input')

    # ... for categorical variables
    cat_inputs = {}
    cat_input_list = []
    embeddings = []
    if cat_encoding == 'integer':
        # Embedding layers are used between each input categorical variables and the network
        def _create_embeddings(SH_EMB, CAT_VARS):
#             for var_idxs in SH_EMB:
#                 var1 = CAT_VARS[var_idxs[0]]
#                 _dim = cat_emb_dim if type(cat_emb_dim) is int else cat_emb_dim[var1]
#                 embedding = Embedding(input_dim=len(cat_levels[var1]), output_dim=_dim, input_length=1)
#                 for var in [CAT_VARS[j] for j in var_idxs]:
#                     cat_inputs[var] = Input(shape=(1,), dtype='int32', name='cat_input_{}'.format(len(cat_inputs)))
#                     embeddings.append(Flatten()(embedding(cat_inputs[var])))

            # Flatten the dictionary into a list
#             shared_embs = [CAT_VARS[item] for sublist in SH_EMB for item in sublist]
            for var in CAT_VARS:
#                 if var not in shared_embs:
                print(var)
                _dim = cat_emb_dim if type(cat_emb_dim) is int else cat_emb_dim[var]
                print(var)
                print(_dim)
                embedding = Embedding(input_dim=len(cat_levels[var]), output_dim=_dim, input_length=1)
                cat_inputs[var] = Input(shape=(1,), dtype='int32', name='cat_input_{}'.format(len(cat_inputs)))
                embeddings.append(Flatten()(embedding(cat_inputs[var])))

        _create_embeddings(SHARED_EMBEDDING, CATEGORICAL_COLS)
        if include_x_vars:
            _create_embeddings(X_SHARED_EMBEDDING, X_CATEGORICAL_COLS)

        emb_input = concatenate([num_bin_input] + embeddings)
        for var in cat_vars:
            cat_input_list.append(cat_inputs[var])

    else:
        # Hashing or one-hot encoding is used, so the categorical variables are already encoded and will be directly
        # fed into the network (i.e., w/o intermediary embedding layers)
        cat_input_list.append(Input(shape=(dim_cat_var,), dtype='float32', name='cat_input_0'))
        emb_input = concatenate([num_bin_input] + cat_input_list)

    def _activation(x):
        if activation == 'prelu':
            return PReLU()(x)
        else:
            return Activation(activation)(x)

    # There are approx. 500K samples. Too few. According to Ilya Sutskever,
    # (http://yyue.blogspot.no/2015/01/a-brief-overview-of-deep-learning.html), to avoid overfitting we should not have
    #  more than 500K/32 parameters (approx. 18K). Too few!
    x = Dense(128)(emb_input)
    # x = BatchNormalization()(x)
    # x = _activation(x)
    # x = Dropout(0.2)(x)
    # x = Dense(64)(x)
    # x = BatchNormalization()(x)
    # x = _activation(x)
    # x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = Model([num_bin_input] + cat_input_list, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure, gmean])

    return model


def precision(y_true, y_pred):
    """Precision metric.        
        
    Only computes a batch-wise average of precision.        
        
    Computes the precision, a metric for multi-label classification of        
    how many selected items are relevant.        
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.        
        
    Only computes a batch-wise average of recall.        
        
    Computes the recall, a metric for multi-label classification of        
    how many relevant items are selected.        
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# def fbeta_score(y_true, y_pred, beta=1):
#     """Computes the F score.
#
#     The F score is the weighted harmonic mean of precision and recall.
#     Here it is only computed as a batch-wise average, not globally.
#
#     This is useful for multi-label classification, where input samples can be
#     classified as sets of labels. By only using accuracy (precision) a model
#     would achieve a perfect score by simply assigning every class to every
#     input. In order to avoid this, a metric should penalize incorrect class
#     assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
#     computes this, as a weighted mean of the proportion of correct class
#     assignments vs. the proportion of incorrect class assignments.
#
#     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
#     correct classes becomes more important, and with beta > 1 the metric is
#     instead weighted towards penalizing incorrect class assignments.
#     """
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')
#
#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0
#
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score
#

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.        
        
    Here it is only computed as a batch-wise average, not globally.        
    """
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * p * r / (p + r + K.epsilon())
    return f_score


def gmean(y_true, y_pred):
    """Computes the g-measure, the geometric mean of precision and recall.        

    Here it is only computed as a batch-wise average, not globally.        
    """
    # If there are no true positives, fix the G score at 0
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    g_mean = K.sqrt(p*r)
    return g_mean


def train_and_evaluate_model(model, data_train, data_test, nb_epochs=100,
                             batch_size=32, cvscores=None, verbose=False, chkp_file=None,
                             earlystop_pat=10,
                             r_balanced_batch=1, oversample = True, apply_class_weights=False,
                             smooth_factor=0.1):

    # Define callbacks for early stopping and model checkpointing
    earlystopping = EarlyStopping(monitor='val_loss', patience=earlystop_pat, verbose=True, mode='auto')
    checkpoint = ModelCheckpoint(chkp_file, monitor='val_acc', verbose=verbose, save_best_only=True, mode='max')

    class_weights = {0: 1.0, 1:1.0}
    if apply_class_weights:
            y = data_train["labels"].values
            class_weights = get_class_weights(y.reshape((1, y.shape[0]))[0], smooth_factor=smooth_factor)
    print("Using class_weights: ", class_weights)
    
    temp = []
    for col in data_test["cat"].columns:
        temp.append(data_test["cat"][col])
    data_test_cat = pd.concat(temp, axis=1)

    # Fit the model
    if oversample:
        # correct shapes
        
        print("Oversampling with ration neg/pos=", r_balanced_batch)
        gen = OverSamplingBatchGenerator(data_train, batch_size=batch_size, r=r_balanced_batch)

        history = model.fit_generator(gen.generator(),
                                  validation_data=(
                                      [data_test["num"].values]+[data_test["cat"].values],
                                      data_test["labels"].values),
                                  steps_per_epoch=gen.get_no_batches(),
                                  epochs=nb_epochs,
                                  verbose=verbose,
                                  max_q_size=10,
                                  workers=1,
                                  pickle_safe=True,class_weight=class_weights,
                                  callbacks=[checkpoint, earlystopping])

    else:
        # right shapes
        history = model.fit([data_train["num"].values] +
                            [data_train["cat"].values],
                            data_train["labels"].values,
                            validation_data=(
                                [data_test["num"].values]+[data_test["cat"].values],
                                      data_test["labels"].values),
                            shuffle=True,
                            epochs=nb_epochs, batch_size=batch_size, class_weight=class_weights,
                            verbose=verbose, callbacks=[checkpoint,earlystopping])


    pickle.dump(history.history, open("history.pickle", "wb"))
    
    # model.save('model.h5')
    # print("saved model")

    # summarize history for accuracy
#     plt.ioff()
#     fig = plt.figure()
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./accuracy.png')
#     plt.close(fig)
    # plt.show()

    # summarize history for loss
#     fig = plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('./loss.png')
#     plt.close(fig)
    #plt.show()


    #print("%s: %.2f%%" % (model.metrics_names[1], history[-1][1] * 100))
    if cvscores:
        cvscores.append(history[-1][1] * 100)

def get_class_weights(y, smooth_factor=0):
        """
        Returns the weights for each class based on the frequencies of the samples
        :param smooth_factor: factor that smooths extremely uneven weights
        :param y: list of true labels (the labels must be hashable)
        :return: dictionary with the weight for each class
        """
        counter = Counter(y)

        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p

        majority = max(counter.values())

        return {cls: float(majority / count) for cls, count in counter.items()}

# The dataset is highly unbalanced. This function generates balanaced batches using the oversampling method proposed in
# Yilin Yan, Min Chen, Mei-Ling Shyu and Shu-Ching Chen, "Deep Learning for Imbalanced Multimedia Data Classification"
# https://users.cs.fiu.edu/~chens/PDF/ISM15.pdf
#
#  OBS!
#  batch_size must be divisible by (1 + r). r = n_batch_neg / n_batch_pos

# read about balancing datasets
class OverSamplingBatchGenerator:
    def __init__(self, data_train, batch_size=32, r=1):
        # n_batch = n_batch_neg + n_batch_pos
        # r = n_batch_neg / n_batch_pos
        # N = floor(n_neg / n_batch_neg) = floor(n_neg * (1 + r) / (n_batch * r))

       # if (batch_size % (1 + r)) <> 0:
       #     raise Exception("batch_size must be divisible by (1 + r)")

        self.data_train = data_train
        self.neg_idx = (data_train["labels"][data_train["labels"][LABEL_COL] == 0]).index.values
        self.pos_idx = (data_train["labels"][data_train["labels"][LABEL_COL] == 1]).index.values
        # Number of positive and negative examples per batch
        self.n_batch_pos = int(batch_size / (1 + r))
        self.n_batch_neg = (batch_size - self.n_batch_pos)
        # Total number of negative examples
        n_neg = self.neg_idx.size
        # Number of batches
        self.N = int((n_neg * (1 + r) / (batch_size * r)))
        if self.N*self.n_batch_neg > n_neg:
            self.N = int(n_neg/self.n_batch_neg)

    def get_no_batches(self):
        return self.N

    def generator(self):
        labels = np.vstack([np.ones((self.n_batch_pos, 1)), np.zeros((self.n_batch_neg, 1))])
        while True:
            np.random.shuffle(self.neg_idx)
            # Shuffle negative data at the beginning of each epoch. There should be self.N steps per epoch
            # (i.e., one complete for-loop).
            # No need to shuffle positive data, since we randomly sample from it for every batch
            
            for start_idx in range(0, self.N * self.n_batch_neg, self.n_batch_neg):
                batch_pos_idx = np.random.choice(self.pos_idx, self.n_batch_pos, replace=False)
                batch_neg_idx = self.neg_idx[start_idx: (start_idx + self.n_batch_neg)]
                batch = [np.vstack([self.data_train["num"].loc[batch_pos_idx].values,
                                    self.data_train["num"].loc[batch_neg_idx].values])] + \
                [np.vstack([self.data_train["cat"].loc[batch_pos_idx].values,
                            self.data_train["cat"].loc[batch_neg_idx].values])]
                batch.append(labels)
                batch = shuffle(*batch)
                
                yield (batch[0:-1], batch[-1])

def load_and_preprocess_data(args):
    fname_num = "handset_num_{}{}{}{}{}{}_{}.h5".format(int(args.std), int(args.pca_whiten),
                                                        int(args.log_xform), int(args.binary_enc),
                                                        int(args.x_vars),
                                                        args.pca_reduce, args.data_split_id)
    fname_cat = "handset_cat_{}{}{}_{}.h5".format(args.cat_enc, int(args.binary_enc), int(args.x_vars),
                                                  args.data_split_id)

    df = load_data()
    
    df['CU_MAP_SEGMENT'].fillna(0, inplace=True)
    df['CU_MAP_SEGMENT'] = df['CU_MAP_SEGMENT'].astype(float).astype(int)
    df['CU_MAP_SEGMENT'] = df['CU_MAP_SEGMENT'].astype(str)
    
    df['MPP_CHURN_SEGMENT_ADJ'].fillna(0, inplace=True)
    df['MPP_CHURN_SEGMENT_ADJ'] = df['MPP_CHURN_SEGMENT_ADJ'].astype(float).astype(int)
    df['MPP_CHURN_SEGMENT_ADJ'] = df['MPP_CHURN_SEGMENT_ADJ'].astype(str)
    
    df_labels = pd.DataFrame(df[LABEL_COL].astype(int))

    print("generating dictionary with levels of catagorical variables...")
    cat_vars = CATEGORICAL_COLS + X_CATEGORICAL_COLS if args.x_vars else CATEGORICAL_COLS
    cat_levels = get_distinct_values(df, cat_vars, args.x_vars)


    save = False
    new_split = True
    if args.data_split_id > 0:
        save = True
        fsplit = "split_{}.h5".format(args.data_split_id)
        if os.path.isfile(fsplit):
            print ("Reusing data split with id={}".format(args.data_split_id))
            new_split = False
            h5f = h5py.File(fsplit, 'r')
            train = h5f['train'][:]
            h5f.close()

    if new_split:
        print ("Generating new data split with id={}".format(args.data_split_id))
        ssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        splits = ssplit.split(df.values, df_labels)
        train, _ = next(splits)
        if save:
            print ("Saving split to disk...")
            h5f = h5py.File(fsplit, 'w')
            h5f.create_dataset('train', data=train)
            h5f.close()

    df_train_num = df_test_num = df_train_cat = df_test_cat = df_train_labels = df_test_labels = None
    data_missing = ['num', 'cat']
    if os.path.isfile(fname_num):
        print("Loading previously pre-processed numerical data...")
        df_train_num = pd.read_hdf(fname_num, "train_num")
        df_test_num = pd.read_hdf(fname_num, "test_num")
        df_train_labels = pd.read_hdf(fname_num, "train_labels")
        df_test_labels = pd.read_hdf(fname_num, "test_labels")
        data_missing.remove('num')

    if os.path.isfile(fname_cat):
        print("Loading previously pre-processed categorical data...")
        df_train_cat = pd.read_hdf(fname_cat, "train_cat")
        df_test_cat = pd.read_hdf(fname_cat, "test_cat")
        data_missing.remove('cat')

    if len(data_missing) > 0:
        print("Preprocessing training data...")
        _df_train_num, _df_train_cat, mean, std, pca = preprocess_data(df.loc[train],
                                                                                 cat_levels,
                                                                                 data_to_process=data_missing,
                                                                                 cat_encoding=args.cat_enc,
                                                                                 cat_emb_dim=CAT_EMB_DIM,
                                                                                 standardize=args.std,
                                                                                 pca_whiten=args.pca_whiten,
                                                                                 pca_reduce=args.pca_reduce,
                                                                                 log_transform=args.log_xform,
                                                                                 binary_enc=args.binary_enc,
                                                                                 include_x_vars=args.x_vars)
        df.drop(df.index[train], inplace=True)

        print("Preprocessing test data...")
        _df_test_num, _df_test_cat, _, _, _ = preprocess_data(df,
                                                            cat_levels,
                                                            data_to_process=data_missing,
                                                            cat_encoding=args.cat_enc,
                                                            cat_emb_dim=CAT_EMB_DIM,
                                                            standardize=args.std,
                                                            pca_whiten=args.pca_whiten,
                                                            pca_reduce=args.pca_reduce,
                                                            log_transform=args.log_xform,
                                                            binary_enc=args.binary_enc,
                                                            include_x_vars=args.x_vars,
                                                            mean=mean, std=std, pca=pca)

        df = None
        if df_train_num is None:
            df_train_num = _df_train_num
            df_test_num = _df_test_num

        if df_train_cat is None:
            df_train_cat = _df_train_cat
            df_test_cat = _df_test_cat

        if df_train_labels is None:
            df_train_labels = df_labels.loc[train]
            df_train_labels.reset_index(drop=True, inplace=True)
            df_labels.drop(df_labels.index[train], inplace=True)
            df_labels.reset_index(drop=True, inplace=True)
            df_test_labels = df_labels

        if save:
            print("Storing preprocessed data...")
            if 'num' in data_missing:
                df_train_num.to_hdf(fname_num, "train_num")
                df_test_num.to_hdf(fname_num, "test_num")
                df_train_labels.to_hdf(fname_num, "train_labels")
                df_test_labels.to_hdf(fname_num, "test_labels")

            if 'cat' in data_missing:
                df_train_cat.to_hdf(fname_cat, "train_cat")
                df_test_cat.to_hdf(fname_cat, "test_cat")
                
    return {"num": df_train_num, "cat": df_train_cat, "labels": df_train_labels}, \
           {"num": df_test_num, "cat": df_test_cat, "labels": df_test_labels}, cat_levels

def main(args):
    # fix random seed for reproducibility
    np.random.seed(SEED)

    if args.cross_val > 0:
        # OBS! Need to fix this part

        n_folds = args.cross_val
        # skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        #
        # cvscores = []
        # i = 1
        # for train, test in skf.split(df_num.values, labels):
        #     print("Running Fold {} of {}".fomat(i, n_folds))
        #     i += 1
        #     model = None # Clearing the NN.
        #     print("Creating model...")
        #     model = create_model(df_num.columns, df_cat.columns, cat_encoding=args.cat_enc, cat_emb_dim = CAT_EMB_DIM,
        #                          cat_levels = cat_levels, include_x_vars=args.x_vars, activation=args.activation)
        #     chkp_file = "handset_weights_fold_{}.best.hdf5".format(i)
        #     print("Training model...")
        #     train_and_evaluate_model(model, (df_num.values[train], df_cat.values[train]), labels[train],
        #                              (df_num.values[test], df_cat.values[test]), labels[test], nb_epochs=args.epochs,
        #                              batch_size=args.batch_size, verbose=args.verbose, chkp_file=chkp_file,
        #                              earlystop_pat=args.earlystop, cvscores=cvscores)
        #
        # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    else:
        data_train, data_test, cat_levels = load_and_preprocess_data(args)
        
        print("Creating model...")
        model = create_model(data_train["num"].columns, data_train["cat"].columns, cat_encoding=args.cat_enc,
                             cat_emb_dim=CAT_EMB_DIM, cat_levels=cat_levels, include_x_vars=args.x_vars,
                             activation=args.activation)

        chkp_file = "handset_weights.best.hdf5"

        print("Training model...")
        train_and_evaluate_model(model, data_train, data_test, nb_epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 oversample=args.oversample,
                                 apply_class_weights=args.apply_class_weights,
                                 smooth_factor=args.smooth_factor,
                                 verbose=args.verbose, chkp_file=chkp_file,
                                 earlystop_pat=args.earlystop)


def bool_arg(string):
        value = string.lower()
        if value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))

if __name__ == "__main__":
    # minimal preprocessing/ tinkering
    # comments relative to handset_model_copy3

    parser = argparse.ArgumentParser()
    # (epochs=10 to shorten wait time)
    parser.add_argument('--epochs', default=10, type=int,
                        help="Nr of epochs. Default is 100", dest="epochs")
    parser.add_argument('--batch_size', default=256, type=int,
                        help="Batch size. Default is 32", dest="batch_size")
    parser.add_argument('--earlystop', default=5, type=int,
                        help="Number of epochs with no improvement after which training will be stopped.",
                        dest="earlystop")
    parser.add_argument('--verbose', default=True, type=bool_arg, help="If True (default), verbose output",
                        dest="verbose")
    parser.add_argument('--cross_val', default=0, type=int,
                        help="Number of folds (if bigger than 0) to use for cross validation. Default is 0.",
                        dest="cross_val")
    # (no applying class weights)
    parser.add_argument('--apply_class_weights', default=False, type=bool_arg,
                        help="If True, apply different loss weights (based on frequency of samples) to different "
                             "classes.",
                        dest="apply_class_weights")
    # (no smooth factor)
    parser.add_argument('--smooth_factor', default=0, type=float,
                        help="Smooth factor to be used when calculating class weights, so that highly unfrequent "
                        "classes do not get huge weights.",
                        dest="smooth_factor")
    parser.add_argument('--oversample', default=True, type=bool_arg,
                        help="If True (default), apply oversampling to generate balanced batches.",
                        dest="oversample")
    parser.add_argument('--ratio', default=1, type=int,
                        help="Ratio of negative to positive samples to use for balanced batch generation "
                             "(if oversample=True)",
                        dest="ratio")
    parser.add_argument('--activation', default='prelu',
                        help="NN activation to be used. Default is prelu",
                        dest="activation")
    parser.add_argument('--x_vars', default=False, type=bool_arg, help="If True (default), include X variables",
                        dest="x_vars")
    parser.add_argument('--std', default=True, type=bool_arg, help="If True (default), standardize data.",
                        dest="std")
    parser.add_argument('--pca_whiten', default=False, type=bool_arg, help="If True (default), PCA-whiten data.",
                        dest="pca_whiten")
    parser.add_argument('--pca_reduce', default=0, type=float,
                        help="{0, 1, 0<x<1} If 0, no dimensionality reduction is done. If 1, Thomas P. Minka's method "
                             "('Automatic Choice of Dimensionality for PCA'. NIPS 2000) is used to determine the "
                             "number of dimensions to keep. If 0 < pca_reduce < 1, enough number of dimensions will "
                             "be kept to keep 'pca_reduce' percentage of variance explained. Default is 0.9.",
                        dest="pca_reduce")
    # (basic one-hot without embeddings)
    parser.add_argument('--cat_enc', default='one-hot',
                        help="Encoding to be used for categorical variables. Default is 'integer' "
                             "(embedding layers will then be used). Other alternatives: 'hashing_char', "
                             "'hashing_all', 'one-hot'.",
                        dest="cat_enc")
    parser.add_argument('--log_xform', default=False, type=bool_arg, help="If True (default), log-transform data.",
                        dest="log_xform")
    # (encode as 1/0)
    parser.add_argument('--binary_enc', default=True, type=bool_arg,
                        help="If False (default), the negative cases of binary variables will be represented as -1 "
                             "instead of 0.", dest="binary_enc")
    # id = 2
    parser.add_argument('--data_split_id', default=1, type=int,
                        help="Id for the train-test data split to be used. If a new id is given, a new data split "
                             "will be generated and saved to disk with the given id. If id is 0 (default), a new "
                             "split will be generated, but not saved to disk. If a previously used id is given, "
                             "a previously generated and saved data split with that id will be used.",
                        dest="data_split_id")
    parser.add_argument("-f")
    args = parser.parse_args()
    main(args)