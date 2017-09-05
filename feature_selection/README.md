# feature_selection

## Data and Preprocessing
* id=1

_note 1: features ending with an underscore followed by lowercase letters or number are categorical features (after one-hot encoding)_

_note 2: the following lists of features are ordered by order-of-appearance in the data rather than a metric_

---

## SelectKBest (sklearn): f_classif (between each feature col and the label col)
### k = 20 (select 20 best):
CU_AGE  
CU_U_NET_REV_AVG_3MO  
CU_U_MB_AVG_3MO  
MPP_MB_LAST1  
MPP_MB_LAST2  
MPP_MB_LAST3  
MPP_MB_SUM_3MO  
MPP_MB_AVG_3MO  
MPP_NO_VOICE_DOM_LAST2  
MPP_NO_VOICE_DOM_LAST3  
MPP_GROSS_PERIODIC_FEE_FULL  
MPP_NET_REVENUE  
CU_MAP_SEGMENT_6  
CLM_LIVSFASE_SEGMENT_ung voksen  
CU_U_MAIN_DEV_OS_TYPE_iphone os  
CU_U_MAIN_DEV_OS_TYPE_android  
CU_U_MAIN_DEV_PRODUCERNAME_apple  
MPP_DEVICE_OS_TYPE_iphone os  
MPP_DEVICE_OS_TYPE_android  
MPP_DEVICE_PRODUCERNAME_apple  

---

## Correlation (between each feature col and the label col)
### |corr| > 0.04:

feature col | abs(corr)
--- | ---
CU_AGE | 0.0432861057372
CU_U_MB_AVG_3MO | 0.0455189620728
MPP_MB_SUM_3MO | 0.0425954516754
MPP_MB_AVG_3MO | 0.0424083809852
MPP_GROSS_PERIODIC_FEE_FULL | 0.0489477804739
CU_MAP_SEGMENT_6 | 0.0464689746653
CLM_LIVSFASE_SEGMENT_ung voksen | 0.04190183543
CU_U_MAIN_DEV_OS_TYPE_iphone os | 0.0541875972373
CU_U_MAIN_DEV_PRODUCERNAME_apple | 0.0541875972373
MPP_DEVICE_OS_TYPE_iphone os | 0.042598699569
MPP_DEVICE_PRODUCERNAME_apple | 0.042598699569

---

The features below are chosen as follows:
```py
# pseudocode
X0 = data with label 0
X1 = data with label 1
for group in [X0, X1]:
  metric[i] = calculate metric for feature i in group
  # metric = mean or variance or median
for i in all_features:
  diff = (metric[i] from X1) - (metric[i] from X0)
set a threshold for diff and store the largest diffs in a dictionary
# features showing significantly different behavior (in terms of a metric) from being in X0 to being in X1 should be able to contribute to predicting the different labels 0/1
```

---

## Mean
### |meanDiff| > 0.5:

feature col | abs(meanDiff)
--- | ---
CU_AGE | 0.614539254253
CU_U_NET_REV_AVG_3MO | 0.514587359183
CU_U_MB_AVG_3MO | 0.641625300575
MPP_MB_LAST1 | 0.521722536441
MPP_MB_LAST2 | 0.551535217464
MPP_MB_LAST3 | 0.567267376188
MPP_MB_SUM_3MO | 0.60158436149
MPP_MB_AVG_3MO | 0.599217503742
MPP_GROSS_PERIODIC_FEE_FULL | 0.693419478547
MPP_NET_REVENUE | 0.56731436173
CU_MAP_SEGMENT_6 | 0.55108643158
CU_U_MAIN_DEV_OS_TYPE_iphone os | 0.761851886577
CU_U_MAIN_DEV_OS_TYPE_android | 0.505456057002
CU_U_MAIN_DEV_PRODUCERNAME_apple | 0.761851886577
MPP_DEVICE_OS_TYPE_iphone os | 0.604549872621
MPP_DEVICE_OS_TYPE_android | 0.547848345531
MPP_DEVICE_PRODUCERNAME_apple | 0.604549872621

* as we can see, there is significant overlap between the meanDiff features and the f_classif features

## Variance
### |varDiff| > 1:

feature col | abs(varDiff)
--- | ---
CU_MPR_NO_MMS_DOM_LAST1 | 1.42333386142
CU_MPR_NO_SMS_INT_LAST1 | 1.06722139832
CU_FIX_NO_VOICE_INT_LAST2 | 2.49542246679
CU_FIX_NO_VOICE_INT_LAST3 | 3.05030083975
CU_U_MB_AVG_3MO | 1.56430575587
MPP_BANKID_USED_LAST1 | 1.57393625331
MPP_BANKID_USED_LAST2 | 1.54462117929
MPP_BANKID_USED_LAST3 | 1.55374375763
MPP_MB_LAST1 | 2.3390980876
MPP_MB_LAST2 | 1.48972499955
MPP_MB_LAST3 | 1.71293747125
MPP_MB_SUM_3MO | 1.65071812047
MPP_MB_AVG_3MO | 1.65009006214
MPP_KR_SMS_INT_LAST3 | 2.54310807937
MPP_NO_VOICE_DOM_LAST3 | 1.00714918042
MPP_NO_VOICE_INT_LAST1 | 1.17235679347
MPP_NO_VOICE_INT_LAST2 | 1.11125668268
MPP_NO_VOICE_INT_LAST3 | 1.12647016259
MPP_NET_OTHER_FEE | 1.30841506924

## Median
### |medDiff| > 0.5:

feature col | abs(medDiff)
--- | ---
CU_AGE | 0.60820436842
HH_ANT_VOKSEN | 1.20883448015
CU_U_MB_AVG_3MO | 0.501910253248
CU_U_MAIN_DEV_MODEL_ID | 0.79163023219
MPP_GROSS_PERIODIC_FEE_FULL | 0.52300250517
MPP_NET_REVENUE | 0.525811311234
CU_GENDER_m | 2.0
CU_GENDER_k | 2.0
CU_ADSL_OK_RESULT_verify | 2.0
CU_U_MAIN_DEV_OS_TYPE_iphone os | 2.0
CU_U_MAIN_DEV_PRODUCERNAME_apple | 2.0
CU_U_MAIN_DEV_CATEGORY_smartphone lte | 2.0
MPP_DEVICE_CATEGORY_smartphone lte | 2.0
MPP_BINDING_TYPE_binding terminal | 2.0
