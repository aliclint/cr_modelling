import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def read_split_data(file_path, test_size = 0.2, random_state = 20210210, label_col_name = 'SeriousDlqin2yrs'):
    df = pd.read_csv(file_path, index_col = 0, encoding='utf8')
    X_cols = [col for col in df.keys() if col != 'SeriousDlqin2yrs']
    y_cols = ['SeriousDlqin2yrs']
    X = df[X_cols]
    y = df[y_cols]
    X_train, X_test, y_train, y_test = train_test_split(X ,y ,
                                                        test_size=test_size, random_state=20210210)
    return X_train, X_test, y_train, y_test

def generate_eda_plots(df):
    pass


def correlation_heatmap(df):
  corr = df.corr()
  return sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)


def find_missing_columns(df):
  return df.columns[df.isnull().any()].tolist()


def feature_processing(X_train, X_test, y_train):
    bins = pd.IntervalIndex.from_tuples(
        [(-1, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, X_train.age.max())])
    X_train['age_group'] = create_group(X_train, 'age', bins, None)
    X_test['age_group'] = create_group(X_test, 'age', bins, None)
    X_train = outlier_processing(X_train)
    X_test = outlier_processing(X_test)

    X_train['NumberOfDependents'] = missing_value_processing(X_train, 'NumberOfDependents', 0)
    X_test['NumberOfDependents'] = missing_value_processing(X_test, 'NumberOfDependents', 0)

    X_train['MonthlyIncome'] = missing_value_processing(X_train, 'MonthlyIncome', X_train['MonthlyIncome'].median())
    X_test['MonthlyIncome'] = missing_value_processing(X_test, 'MonthlyIncome', X_test['MonthlyIncome'].median())

    X_train_feature_selection, X_test_feature_selection = feature_selection(X_train, X_test, y_train)

    return X_train_feature_selection, X_test_feature_selection


def create_group(df, col, bins, labels):
    output = pd.cut(df[col], bins=bins, labels=labels).cat.codes
    return output


def feature_selection(X_train, X_test, y):
  clf = ExtraTreesClassifier(n_estimators=50)
  clf = clf.fit(X_train, y)
  model = SelectFromModel(clf, prefit=True)
  X_train_output = model.transform(X_train)
  X_train_output = pd.DataFrame(X_train[X_train.columns[model.get_support()]])
  X_test_output = pd.DataFrame(X_test[X_test.columns[model.get_support()]])

  return X_train_output, X_test_output


def missing_value_processing(df, col, value):
  return df[col].fillna(value)


def outlier_processing(df):
  transformer = RobustScaler().fit_transform(df)
  output_df = pd.DataFrame(transformer)
  output_df.columns = df.columns
  return output_df