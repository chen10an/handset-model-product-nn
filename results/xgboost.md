# XGBoost

## Data and Preprocessing
* id=2

## Model
```py
npos = len([i for i in y_train if i == 1])
nneg = len([i for i in y_train if i == 0])

from xgboost.sklearn import XGBClassifier
xgb1 = XGBClassifier(learning_rate=0.1,
                     n_estimators=30,
                     max_depth=5,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='binary:logistic',
                     seed=0,
                     scale_pos_weight=nneg/npos,
                     silent=False
                    )
```

all params:
{'base_score': 0.5,
 'colsample_bylevel': 1,
 'colsample_bytree': 0.8,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 5,
 'min_child_weight': 1,
 'missing': None,
 'n_estimators': 30,
 'nthread': -1,
 'objective': 'binary:logistic',
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 199.61564918314704,
 'seed': 0,
 'silent': False,
 'subsample': 0.8}

These starting params come from:
* https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
* https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
* https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md

## Results

[XGB] | train | eval
--- | --- | ---
auc | 0.790120 | 0.748477
accuracy | 0.754290 | 0.755064
precision | 0.016544 | **0.014937**
recall | 0.826311 | 0.741824

train-confusion-matrix:

[XGB] | predicted - | predicted +
--- | --- | ---
actual - | 350054 | 114252
actual + | 404 | 1922

test-confusion-matrix:

[XGB] | predicted - | predicted +
--- | --- | ---
actual - | 87654 | 28424
actual + | 150 | 431

train-true-pos-rate: 0.826311	eval-true-pos-rate: 0.741824
