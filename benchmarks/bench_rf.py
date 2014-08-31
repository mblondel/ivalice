import time

import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from ivalice.regression import RFRegressor

data = fetch_covtype(download_if_missing=True, shuffle=True, random_state=0)
X, y = data.data, data.target

n_samples = 10000
mask = y <= 2
Xb = X[mask][:n_samples]
yb = y[mask][:n_samples]

Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, train_size=0.75,
                                              test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators=100,
                           max_depth=3,
                           max_features=0.6)
start = time.time()
rf.fit(Xb_tr, yb_tr)
print "RandomForestRegressor"
print time.time() - start, "seconds"
y_pred = rf.predict(Xb_te)
print mean_squared_error(yb_te, y_pred)
print

rf = RFRegressor(n_estimators=100,
                 max_depth=3,
                 max_features=0.6)
start = time.time()
rf.fit(Xb_tr, yb_tr)
print "RandomForestRegressor"
print time.time() - start, "seconds"
y_pred = rf.predict(Xb_te)
print mean_squared_error(yb_te, y_pred)
