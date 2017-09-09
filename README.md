# handset-model-product-nn
Telenor Handset Model with Product-based Neural Networks

This repository explores a binary classification problem through...
1. feature selection
2. xgboost
3. wide&deep (Google)
4. complementary neural networks (cmtnn)
5. **product-based neural networks (pnn)** (main focus)

_note 1: the relevant papers are included in the papers directory_

_note 2: the company data has not been included in this repository_

For an **overview**, refer to the presentation directory.

The specific **results** obtained from different models are in the results directory.

---

**label column:** "TARGET_S_TO_S_APPLE"

### Split IDs and their contents (produced via handset_model_current.py):
* id=1: all features
  * dropped: ['Unnamed: 0', 'ID', 'MPP_NET_DISCOUNT_OTHER_FEE']
  * categorical and binary encoded as -1/1
  * standardized numerical

standardized numerical and categorical/binary encoded as 0/1:
* id=2: sklearn selectKBest w/ f_classif features
* id=3: all cat features
* id=4: all features selected in feature_selection directory
* id=5: f_classif features, cat vars without any encoding
* id=6: features from COLS_TERM (look in handset_model_current.py)

note: refer to features in the feature_selection directory

---

note: handset_model_original.py was written for python 2.7 while handset_model_current.py has been debugged (and further modified) for python 3.6
