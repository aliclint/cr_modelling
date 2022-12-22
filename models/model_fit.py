
from features.build_features import *
from models.classifier import *
import pickle

def run_save_model():
    file_path = 'data//cs-training.csv'
    X_train, X_test, y_train, y_test = read_split_data(file_path)
    X_train_fs, X_test_fs = feature_processing(X_train, X_test, y_train.values.ravel())
    y_pred_cv, clf_cv = fit_rf_gs_cv(X_train_fs, X_test_fs, y_train.values.ravel())
    pickle.dump(clf_cv, open(filename, 'wb'))