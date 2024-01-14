import pandas as pd

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

from scripts import functions

dataset = "breast_cancer"
raw_df = pd.read_csv(f"ML-datasets/{dataset}.csv")

print(raw_df.head())
print(raw_df.info())
print(raw_df.describe(include="all"))

raw_df = functions.nan_string_character(raw_df, "?")
raw_df = functions.replace_nans(raw_df)

#null -> zatím nic
#dtypes -> target variable
class_mapping = {"diabetes": {'positive': 1, 'negative': 0},
               "breast_cancer":{'benign': 0, 'malignant': 1}}
raw_df['Class'] = raw_df['Class'].map(class_mapping[dataset]).astype(int)

# vybrat feature engineering
corr = True
if corr:
  X, y = functions.correlation_filter(raw_df)
else:
  X, y = functions.pca_features(raw_df)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size, stratify=y)

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

functions.plot_metrics(X_train, y_train, X_test, y_test, models=["KNeighborsClassifier", "AdaBoostClassifier", "RandomForestClassifier"])
