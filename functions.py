import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def nan_string_character(df, char="?"):
    # iterate over columns
    for col in df.columns:
        # check string data type
        if df[col].dtype == "O":
            # check char present
            if char in df[col].unique():
                # replace char with NaN
                df[col] = df[col].replace(char, np.NaN)
                try:
                    # try to retype to float
                    df[col] = df[col].astype(float)
                    print(f"converted {col} to numbers")
                except:
                    print(f"not converted {col} to numbers")
    return df


def replace_nans(df, treshold=20):
    for col in df.columns:
        nan_percentage = df[col].isnull().sum() / df.shape[0] * 100
        if treshold > nan_percentage > 0:
            df[col] = df[col].fillna(value=df[col].median())
        elif nan_percentage == 0:
            pass
        else:
            print("Pozor -> hodně NaN")

    return df


def scale_data(df_X, method="StandardScaler"):
    if method == "StandardScaler":
        scaler = StandardScaler()

    scale = scaler.fit(df_X)
    X_scaled = scaler.transform(df_X)

    return pd.DataFrame(X_scaled, columns=df_X.columns)


def split_xy(df, target_col_name="Class"):
    df_split = df.copy()
    Y = df_split.pop(target_col_name).values
    X = df_split
    return X, Y


def correlation_filter(df, treshold=0.2):
    # spočítat korelaci
    corr = df.corr()
    # nakreslit korelační matici
    corr.style.background_gradient(cmap='coolwarm')
    # aplikovat treshold -> výběr nejlepších featur
    corr["Value"] = corr.apply(lambda row: True if row["Class"] > abs(treshold) else False, axis=0)
    # filtrování dat podle nejlepších featur
    valuable_cols = list()
    for i in range(len(df.columns)):
        if corr["Value"].values[i]:
            valuable_cols.append(df.columns[i])
    # vyfiltrovaný df
    df_filtered = df[valuable_cols]

    # split to features and targets
    X, y = split_xy(df_filtered)

    # scale features
    X_scaled = scale_data(X)

    return X, y


def pca_features(df, treshold=80):
    # split to features and targets
    X, y = split_xy(df)

    # scale features
    X_scaled = scale_data(X)

    # perform PCA on feature data
    pca = PCA(n_components=X.shape[1])
    PC = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(data=PC)

    # find how many features to satisfy treshold
    for i in range(1, X.shape[1]):
        exp_variance = sum(list(pca.explained_variance_ratio_ * 100)[:i])
        if exp_variance > treshold:
            break
    # return filtered scaled feature data
    return df_pca.iloc[:, :i], y


def fine_tune_best_model(X_train, y_train, model_name="KNeighborsClassifier"):
    # define which hyperparameters to tune per algorithm
    hyperparameters = {
        "KNeighborsClassifier": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],
                                 'metric': ['minkowski', 'manhattan', 'euclidean']},
        "AdaBoostClassifier": {'n_estimators': [10, 50, 100], 'learning_rate': [0.5, 1, 1.5, 3],
                               'algorithm': ['SAMME', 'SAMME.R']},
        "RandomForestClassifier": {'criterion': ["gini", "entropy"], 'min_samples_leaf': [1, 5, 10],
                                   'n_estimators': [10, 50, 100]}
    }

    # chech whether we have model in the "database" and init instance
    if model_name == "KNeighborsClassifier":
        model = KNeighborsClassifier()
    elif model_name == "AdaBoostClassifier":
        model = AdaBoostClassifier()
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        print("zatím neznám")
        return None

    # create gridseach with defined hyperparameters
    print("Počítám....")

    gs = GridSearchCV(model, hyperparameters[model_name], scoring='f1_weighted', verbose=0)
    gs.fit(X_train, y_train)

    print(f"For {model_name} best parameters: {gs.best_params_}")

    # return best model
    return gs.best_estimator_


def evaluate_model(X_test, y_test, model):
    metrics = {"accuracy": 0,
               "precision": 0,
               "recall": 0,
               "f1": 0}

    y_pred = model.predict(X_test)

    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["precision"] = precision_score(y_test, y_pred)
    metrics["recall"] = recall_score(y_test, y_pred)
    metrics["f1"] = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm


def plot_metrics(X_train, y_train, X_test, y_test, models):
    # here store metrics to given models
    metric_comparison = dict()
    # iterate over models, find best, apply on test and calculate metrics
    for model in models:
        final = fine_tune_best_model(X_train, y_train, model_name=model)
        metrics, cm = evaluate_model(X_test, y_test, final)
        metric_comparison[model] = metrics

    # transform dict with metrics into dataframe for easy plotting
    df = pd.DataFrame.from_dict(
        metric_comparison,
        orient='index',
        columns=list(metrics.keys())
    )

    melted = pd.melt(df.reset_index(), id_vars='index', var_name='metric')
    # rename index to algorithm
    melted = melted.rename(columns={"index": "algorithm"})
    # plot comparison
    sns.barplot(x='metric', y='value', hue='algorithm', data=melted)
    plt.show()
