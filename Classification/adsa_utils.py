"""
Created on Wednesday Oct 16 11:42:55 2024

@author: aurelia power
"""

import re, pandas as pd, numpy as np, matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates

from itertools import combinations

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import RocCurveDisplay
#!pip install kds
#import kds
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

############################################################################
#################### GENERAL VISUALISATION #################################
############################################################################

"""
function that outputs 2d scatter plots for all combination of attributes;
it takes in the data, the names of the regular attributes;
optionally, you can specify the color column, e.g., cluster column, or class column
"""
def scatter_2d_matrix(data, reg_cols, color_col=None):
    indices_combinations = list(combinations(range(0, len(reg_cols)), 2))
    plt.figure(figsize=(10, 10))
    for pair in enumerate(indices_combinations):
        plt.subplot(len(reg_cols)//2 + 1, 2, pair[0] + 1)
        if color_col:
            sns.scatterplot(data, x=reg_cols[pair[1][0]],
                            y=reg_cols[pair[1][1]], hue=color_col)
        else:
            sns.scatterplot(data, x=reg_cols[pair[1][0]],
                            y=reg_cols[pair[1][1]])
    plt.tight_layout()
    plt.show()


############################################################################
############################## CLUSTERING ##################################
############################################################################

"""
    Plots the silhouette scores for the specified clustering algorithm and range of cluster sizes.
    
    Parameters:
    - data: The dataset (numpy array or pandas DataFrame).
    - algorithm: The clustering algorithm (e.g., AgglomerativeClustering or KMeans) instantiated without the number of clusters.
    - cluster_range: List of integers representing the number of clusters to evaluate.
"""
def plot_silhouettes(data, algorithm, cluster_range):
    # Create a figure with subplots to accommodate the number of cluster sizes
    num_plots = len(cluster_range)
    n_rows = (num_plots + 2) // 3  # Adjust rows based on the number of plots
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    # Flatten the axes array for easier iteration
    axs = axs.flatten()
    # Loop through the specified number of clusters
    for i, n_clusters in enumerate(cluster_range):
        # Create an instance of the clustering algorithm with the specified number of clusters
        model = algorithm.set_params(n_clusters=n_clusters)
        # Fit the model and obtain cluster labels
        cluster_labels = model.fit_predict(data)
        # Compute the silhouette scores for each sample
        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        y_lower = 10
        axs[i].set_xlim([-0.1, 1])
        axs[i].set_ylim([0, len(data) + (n_clusters + 1) * 10])
        # Plot silhouette scores for each cluster
        for j in range(n_clusters):
            # Aggregate silhouette scores for samples in cluster j
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()
            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = cm.nipy_spectral(float(j) / n_clusters)
            axs[i].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
            # Label the silhouette plots with cluster numbers
            axs[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
            # Update the y_lower for the next plot
            y_lower = y_upper + 10  # 10 for the space between clusters
        axs[i].set_title(f"Silhouette plot for {n_clusters} clusters")
        axs[i].set_xlabel("Silhouette coefficient values")
        axs[i].set_ylabel("Cluster label")
        # Draw a vertical line for average silhouette score
        axs[i].axvline(x=silhouette_avg, color="red", linestyle="--")
        axs[i].set_yticks([])  # Clear the y-axis labels
        axs[i].set_xticks(np.arange(-0.1, 1.1, 0.2))
    # Remove extra subplots if any
    for ax in axs[num_plots:]:
        fig.delaxes(ax)
    # Adjust layout
    plt.tight_layout()
    plt.show()

############################################################################
############################## CLASSIFICATION ##################################
############################################################################

## function to plot the confusion matrix given the true labels and the predicted labels
def plot_confusion_matrix(y_test, predictions):
  # get and output the confusion matrix
  confusion_m = confusion_matrix(y_test, predictions)
  # can also visualise it
  labels = np.unique(y_test)
  confusion_frame = pd.DataFrame(confusion_m,
                            index=labels, columns=labels)
  plt.figure(figsize=(5, 5))
  sns.heatmap(data=confusion_frame, annot=True, square=True,
              cbar=False)
  plt.xlabel('Predicted Labels')
  plt.ylabel('Actual Labels')
  plt.show()


"""
function to output the average scores and their stds for accuracy, precision, recall, and f-measure metrics;
it takes as arguments the classifier, the regular attributes data, the label column data;
it defines 1 default parameter: the number of folds that's set to 5
"""
def cross_validation_avg_scores(clf, X, y, cv_=5):
  scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
  scores = cross_validate(clf, X, y, scoring=scoring_metrics, groups=y, cv=cv_)
  print(f"Mean accuracy: {scores['test_accuracy'].mean()*100:.2f}% +/-{scores['test_accuracy'].std()*100:.2f}%")
  print(f"Mean precision: {scores['test_precision_macro'].mean()*100:.2f}% +/-{scores['test_precision_macro'].std()*100:.2f}%")
  print(f"Mean recall: {scores['test_recall_macro'].mean()*100:.2f}% +/-{scores['test_recall_macro'].std()*100:.2f}%")
  print(f"Mean F1-score is {scores['test_f1_macro'].mean()*100:.2f}% +/-{scores['test_f1_macro'].std()*100:.2f}%")

## function that combines the 2 above
def custom_crossvalidation(X, y, clf, cv_=5):
  cross_validation_avg_scores(clf, X, y, cv_=cv_)
  predictions = cross_val_predict(clf, X, y, cv=cv_)
  print(classification_report(y, predictions))
  plot_confusion_matrix(y, predictions)


"""
function to generate the test dataset with added columns for predictions and confidence levels;
it takes as arguments an already trained classifier, the regualr attributes data, and the corresponding labels
"""
def get_test_dataset(clf, X_test, y_test):
  confidences = clf.predict_proba(X_test)
  predictions = clf.predict(X_test)
  output = X_test.copy()
  output['true labels'] = y_test
  output['predictions'] = predictions
  for i, class_name in enumerate(clf.classes_):
    output[f'confidence_{class_name}'] = confidences[:, i]
  return output


"""
function to generate and plot multiclass rocs for several classifiers;
it takes as arguments a dictionary of classifiers, the regular data, the label data, the size of the test portion, and the random state
"""
def plot_multiclass_roc(clfs_dict, X, y, test_size_=0.3, random_state_=43):
  # binarize the labels for multi-class ROC
  classes = list(range(len(np.unique(y))))
  labels = label_binarize(LabelEncoder().fit_transform(y), classes=classes)
  n_classes = labels.shape[1]
  X_train, X_test, y_train, y_test = train_test_split(
                                X, labels,
                                test_size=test_size_,
                                stratify=labels,
                                random_state=random_state_)
  # use the One-vs-Rest classier to wrap around each classifier
  y_scores = {}
  # iterate though each classifier, wrap a One-vs-Rest classier around each,
  # train the wrapper and add the resulting probabilities to the scores
  for name, clf in clfs_dict.items():
      clf = OneVsRestClassifier(clf)
      clf.fit(X_train, y_train)
      y_scores[name] = clf.predict_proba(X_test)
  plt.figure(figsize=(7, 5))
  # iterate through each score, compute fprs, tprs, rocs, aucs
  for name, y_score in y_scores.items():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # compute micro-average roc and auc
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
          label=f'{name}: micro-average ROC curve (area = {roc_auc["micro"]:0.2f})')
  # add the diagonal line for random guess
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curves for 3 classifiers on Iris Dataset')
  plt.legend(loc="best")
  plt.show()



############################################################################
############################## REGRESSION ##################################
############################################################################

# function to compute adjusted R2
def adj_rsquared(R2, N, p):
  return 1-((1-R2)*(N-1)/(N-p-1))

# function to compute tolerance for each independent variable
def calculate_tolerance(X):
    tolerance = {}
    for col in X.columns:
        if col == 'const':
            tolerance[col] = ''
            continue
        # Regress the current variable on all others
        other_cols = [c for c in X.columns if c != col]
        model = sm.OLS(X[col], X[other_cols]).fit()
        # Calculate R2 and Tolerance
        r_squared = model.rsquared
        tolerance[col] = 1 - r_squared
    return pd.Series(tolerance)

# crossvalidation function for sklearnregression models
def xval_regress(regressor, X, y, lin=False, return_=False, cv_=5):
  scores = cross_validate(regressor, X, y,
                    scoring=['neg_mean_squared_error', 'explained_variance'],
                    return_estimator=True, cv=cv_)
  AVG_MSE = -scores['test_neg_mean_squared_error'].mean()
  MSE_STD = scores['test_neg_mean_squared_error'].std()
  AVG_RMSE = np.sqrt(AVG_MSE);  RMSE_STD = np.sqrt(MSE_STD)
  AVG_R2 = scores['test_explained_variance'].mean()
  R2_STD = scores['test_explained_variance'].std()
  print(f"Average cross-validated RMSE: {AVG_RMSE:.2f} +/- {RMSE_STD:.2f}")
  print(f"Average cross-validated R-squared: {AVG_R2:.2f} +/- {R2_STD:.2f}")
  if(lin): ## the coef and intercept are only available in linear regression
    print('Coefficients across all folds....')
    for model in scores['estimator']:
      print("Coeficient: ", model.coef_, ", Intercept: ", model.intercept_)
  if(return_):
    return AVG_RMSE, AVG_R2

# function to get only the relevant results from statsmodels OLS
import statsmodels.api as sm
def get_relevant_ols_results(X, y, add_const=True, print_=True):
    if add_const:
        X = sm.add_constant(X, has_constant='add')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    model = sm.OLS(y_train, X_train).fit()
    if print_:
        preds = model.predict(X_test)
        print(f"------- Overall StatsModel OLS Metrics -------")
        print(f"R-squared: {model.rsquared:.2f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.2f} ")
        print(f"AIC: {model.aic:.2f}")
        print(f"RMSE: {root_mean_squared_error(y_test, preds):.2f}")
        print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    df = pd.DataFrame(columns=['coefficient', 'std error', 'std coefficient',
                               't-stat', 'p-value', '0.025', '0.975', 'tolerance'],
                      index=X.columns)
    df['coefficient'] = model.params
    df['std error'] = model.bse
    df['std coefficient'] = df['coefficient'] * (np.std(X_train)/np.std(y_train))
    df['t-stat'] = model.tvalues # can also use df['coefficient'] / df['std error']
    df['p-value'] = model.pvalues
    df['0.025'] = model.conf_int()[0]
    df['0.975'] = model.conf_int()[1]
    df['tolerance'] = calculate_tolerance(X_train).T[X.columns]
    return df.round(2)

# function from crossvalidating a statsmodel OLS
def ols_xval(X, y, cv_=5, output_AIC=False):
  kf = KFold(n_splits=cv_, shuffle=True, random_state=43)
  MAEs = []; RMSEs = []; R2s = [];adj_R2s = []; AICs=[]
  X = sm.add_constant(X, has_constant='add')
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = sm.OLS(y_train, X_train).fit()
    preds = model.predict(X_test)
    MAEs.append(mean_absolute_error(y_test, preds))
    RMSEs.append(root_mean_squared_error(y_test, preds))
    adj_R2s.append(model.rsquared_adj)
    R2s.append(model.rsquared)
    AICs.append(model.aic)
  print(f"Average Cross-validated R-squared: {np.mean(R2s):.2f} +/- {np.std(R2s):.2f}")
  print(f"Average Cross-validated Adjusted R-squared: {np.mean(adj_R2s):.2f} +/- {np.std(adj_R2s):.2f}")
  if(output_AIC):
    print(f"Average Cross-validated AIC: {np.mean(AICs):.2f} +/- {np.std(AICs):.2f}")
  print(f"Average Cross-validated MAE: {np.mean(MAEs):.2f} +/- {np.std(MAEs):.2f}")
  print(f"Average Cross-validated RMSE: {np.mean(RMSEs):.2f} +/- {np.std(RMSEs):.2f}")
  # return the df with relevant resutls for model with split val.
  return get_relevant_ols_results(X, y, add_const=False, print_=False)