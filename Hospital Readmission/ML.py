import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV

############### MACHINE LEARNING ###############
### Feature engenieering and preprocessing ###
seed = 217

# Mapping target variable
df['readmitted'] = df.readmitted.replace({'yes': 1, 'no': 0})

# Create feature and target set
X = df.drop('readmitted', axis=1)
y = df.readmitted

# Splitting feature and target set into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

# Selecting categorical and numeric columns
num_selector = make_column_selector(dtype_exclude='object')
cat_selector = make_column_selector(dtype_include='object')
num_cols = num_selector(X)
cat_cols = cat_selector(X)

# Preprocessing categorical and numeric columns
num_preprocessor = StandardScaler()
cat_preprocessor = OneHotEncoder()

# Merge preprocessed categorical and numeric columns
preprocessor = ColumnTransformer([
    ('StandardScaler', num_preprocessor, num_cols),
    ('One Hot Encoder', cat_preprocessor, cat_cols)
])

# Make a pipelines dictionary
pipelines = {'K-Neighbors Classifier': make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=13)),
             'Logistic Regression': make_pipeline(preprocessor, LogisticRegression(random_state=seed)),
             'Random Forest Classifier': make_pipeline(preprocessor, RandomForestClassifier(random_state=seed))}


# Defining a function to get a table of model metrics.
# Function to get models metrics
def metrics_from_pipes(pipes_dict):
    '''
    This function takes as input a dictionary of ML pipelines  and
    returns a table all the train and test metrics for
    each model in the dictionary
    '''
    train_accs = []
    train_f1s = []
    train_roc_aucs = []
    train_pr_aucs = []
    train_precs = []
    train_recs = []
    train_specs = []
    train_fprs_list = []
    train_fnrs_list = []

    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    test_pr_aucs = []
    test_precs = []
    test_recs = []
    test_specs = []
    test_fprs_list = []
    test_fnrs_list = []

    for name, pipeline in pipes_dict.items():
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        y_probs_test = pipeline.predict_proba(X_test)[:, 1]
        y_probs_train = pipeline.predict_proba(X_train)[:, 1]

        train_precisions, train_recalls, threshold = precision_recall_curve(y_train, y_probs_train)
        test_precisions, test_recalls, threshold = precision_recall_curve(y_test, y_probs_test)

        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train).ravel()
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()

        train_accs.append(accuracy_score(y_train, y_pred_train))
        train_f1s.append(f1_score(y_train, y_pred_train))
        train_roc_aucs.append(roc_auc_score(y_train, y_probs_train))
        train_pr_aucs.append(auc(train_recalls, train_precisions))
        train_precs.append(precision_score(y_train, y_pred_train))
        train_recs.append(recall_score(y_train, y_pred_train))
        train_specs.append(tn_train / (tn_train + fp_train))
        train_fprs_list.append(fp_train / (fp_train + tn_train))
        train_fnrs_list.append(fn_train / (fn_train + tp_train))

        test_accs.append(accuracy_score(y_test, y_pred_test))
        test_f1s.append(f1_score(y_test, y_pred_test))
        test_roc_aucs.append(roc_auc_score(y_test, y_probs_test))
        test_pr_aucs.append(auc(test_recalls, test_precisions))
        test_precs.append(precision_score(y_test, y_pred_test))
        test_recs.append(recall_score(y_test, y_pred_test))
        test_specs.append(tn_test / (tn_test + fp_test))
        test_fprs_list.append(fp_test / (fp_test + tn_test))
        test_fnrs_list.append(fn_test / (fn_test + tp_test))

    # aggregate the performance metric lists into separate dataframes
    train_metrics = pd.DataFrame(
        {'model': pipes_dict.keys(),
         'accuracy': train_accs,
         'f1_score': train_f1s,
         'roc_auc': train_roc_aucs,
         'pr_auc': train_pr_aucs,
         'precision': train_precs,
         'recall': train_recs,
         'specificity': train_specs,
         'false_positive_rate': train_fprs_list,
         'false_negative_rate': train_fnrs_list})

    test_metrics = pd.DataFrame(
        {'model': pipes_dict.keys(),
         'accuracy': test_accs,
         'f1_score': test_f1s,
         'roc_auc': test_roc_aucs,
         'pr_auc': test_pr_aucs,
         'precision': test_precs,
         'recall': test_recs,
         'specificity': test_specs,
         'false_positive_rate': test_fprs_list,
         'false_negative_rate': test_fnrs_list})

    # Merging metrics from train and test set
    train_test_metrics = train_metrics.merge(test_metrics,
                                             on='model',
                                             how='left',
                                             suffixes=('_train', '_test'))

    # Sorting columns
    train_test_metrics = train_test_metrics.reindex(columns=['model',
                                                             'accuracy_train',
                                                             'accuracy_test',
                                                             'f1_score_train',
                                                             'f1_score_test',
                                                             'roc_auc_train',
                                                             'roc_auc_test',
                                                             'pr_auc_train',
                                                             'pr_auc_test',
                                                             'precision_train',
                                                             'precision_test',
                                                             'recall_train',
                                                             'recall_test',
                                                             'specificity_train',
                                                             'specificity_test',
                                                             'false_positive_rate_train',
                                                             'false_positive_rate_test',
                                                             'false_negative_rate_train',
                                                             'false_negative_rate_test'])

    return train_test_metrics.set_index('model').transpose()


# Getting metrics_table
metrics_table = metrics_from_pipes(pipelines)
print('Table 2: Base models metrics table.')
metrics_table.style.background_gradient(cmap='Blues')


# Evaluating multiple models
def boxplot_cv_performances_from_pipes(pipelines_dict):
    results = []

    for pipeline in pipelines_dict.values():
        kf = KFold(n_splits=5)
        cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
        results.append(cv_results)

    # Plot Cross-Validation Performance
    sns.set_style('dark')
    fig = plt.figure(figsize=(10, 8))
    plt.boxplot(results, labels=pipelines.keys(), medianprops={'color': 'mediumseagreen'})
    plt.ylabel('Accurancy score', fontsize=12)

    plt.show()

    print('\n------------------------------------------------------------------')

    # test set performance
    for name, pipeline in pipelines_dict.items():
        pipeline.fit(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        print(f"{name} Test Set Accuracy: {test_score}")


print(
    'Figure 5: Cross-Validation performance for K-Neighbors Classifier, Logistic Regression, Random Forest Classifier.')
boxplot_cv_perfomances_from_pipes(pipelines)


# Defining a function to plot roc_pr_auc_curve.
def roc_pr_auc_curves_from_pipes(pipes_dict):
    fprss = []
    tprss = []
    precs = []
    recs = []
    roc_aucs = []
    pr_aucs = []

    for name, pipeline in pipes_dict.items():
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        fprs, tprs, _ = roc_curve(y_test, y_probs)
        precisions, recalls, _ = precision_recall_curve(y_test, y_probs)

        fprss.append(fprs)
        tprss.append(tprs)
        precs.append(precisions)
        recs.append(recalls)
        roc_aucs.append(roc_auc_score(y_test, y_probs))
        pr_aucs.append(auc(recalls, precisions))

    sns.set_style('white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for i in range(len(fprss)):
        model_name = list(pipes_dict.keys())[i]
        label_model = f"{model_name}\nAUC = {roc_aucs[i]:.3f}"
        ax1.plot(fprss[i], tprss[i], label=label_model)
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Base rate\nAUC = 0.5', color='black')
    ax1.set_xlabel('False positive rate', fontsize=12)
    ax1.set_ylabel('True positive rate', fontsize=12)
    ax1.set_title('Test set area under the ROC curve', fontsize=15)
    ax1.legend(loc="lower right", bbox_to_anchor=(1, 0), ncol=2, frameon=True)

    for i in range(len(precs)):
        model_name = list(pipes_dict.keys())[i]
        label_model = f"{model_name}\nAUC = {pr_aucs[i]:.3f}"
        ax2.plot(recs[i], precs[i], label=label_model)
    ax2.plot([0, 1], [0, 0], linestyle='--', label='Base rate', color='black')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Test set area under the PR curve', fontsize=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, frameon=True)
    plt.subplots_adjust(hspace=0.3)
    sns.despine()


print('Figure 6: Test set area under the ROC curve and PR curve')
roc_pr_auc_curves_from_pipes(pipelines)

# Setting start time to evaluate timing performance
start_time = time.time()

# Create a dictionary if hyper-parameters
param_grid = {'logisticregression__solver': ['liblinear', 'sag', 'saga'],
              'logisticregression__penalty': ['l1', 'l2'],
              'logisticregression__C': [0.1, 1.0, 10],
              'logisticregression__class_weight': [None, 'balanced'],
              'logisticregression__max_iter': [100, 400, 800]}

# Define a cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Instantiate a GridSearchCV object
grid_lr = GridSearchCV(estimator=pipelines['Logistic Regression'],
                       param_grid=param_grid,
                       scoring='accuracy',
                       cv=cv,
                       n_jobs=-1)

# Fit the GridSearchCV object
grid_lr.fit(X_train, y_train)

# Measure the execution time
end_time = time.time()
execution_time = (end_time - start_time) / 60

print(f'The best estimator is: {grid_lr.best_estimator_}\n'
      f'The best params are: {grid_lr.best_params_}\n'
      f'The best score is: {grid_lr.best_score_}\n'
      f'Execution time: {execution_time:.3f} minutes')

# Plot roc_auc_curve fro best estimator and from predictions
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style('white')


# Function to get metrics from tuned model
def metrics_test_from_tuned(model):
    ''' This function take as input a hyper-parameter model
    (i.e GridSearchCV, RandomizedSearchCV) and
    returns a table all the test metrics
    '''
    test_accs = []
    test_f1s = []
    test_roc_aucs = []
    test_pr_aucs = []
    test_precs = []
    test_recs = []
    test_specs = []
    test_fprs_list = []
    test_fnrs_list = []

    mod = model.best_estimator_
    y_pred_test = mod.predict(X_test)

    y_probs_test = mod.predict_proba(X_test)[:, 1]

    test_precisions, test_recalls, threshold = precision_recall_curve(y_test, y_probs_test)

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()

    test_accs.append(accuracy_score(y_test, y_pred_test))
    test_f1s.append(f1_score(y_test, y_pred_test))
    test_roc_aucs.append(roc_auc_score(y_test, y_probs_test))
    test_pr_aucs.append(auc(test_recalls, test_precisions))
    test_precs.append(precision_score(y_test, y_pred_test))
    test_recs.append(recall_score(y_test, y_pred_test))
    test_specs.append(tn_test / (tn_test + fp_test))
    test_fprs_list.append(fp_test / (fp_test + tn_test))
    test_fnrs_list.append(fn_test / (fn_test + tp_test))

    test_metrics = pd.DataFrame(
        {'model': [model.best_estimator_.steps[-1][1].__class__.__name__],
         'accuracy': test_accs,
         'f1_score': test_f1s,
         'roc_auc': test_roc_aucs,
         'pr_auc': test_pr_aucs,
         'precision': test_precs,
         'recall': test_recs,
         'specificity': test_specs,
         'false_positive_rate': test_fprs_list,
         'false_negative_rate': test_fnrs_list})

    return test_metrics.transpose().reset_index().rename(columns={'index': 'metrics', 0: 'values'})


# Getting metrics_table
metrics_table = metrics_test_from_tuned(grid_lr)
print('Table 3: Best estimator metrics table.')
metrics_table


# Plot roc_pr_auc curves from best estimator
def roc_pr_auc_curves_from_best_estimator(model):
    '''This function takes as input an hyper-parameter model and
    returns best estimator ROC_PR AUC curves'''
    precs = []
    recs = []
    pr_aucs = []

    mod = model.best_estimator_
    y_pred = model.predict(X_test)
    y_probs = mod.predict_proba(X_test)[:, 1]

    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    precs.append(precisions)
    recs.append(recalls)
    pr_aucs.append(auc(recalls, precisions))

    sns.set_style('white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    roc_best_estimator = RocCurveDisplay.from_estimator(mod, X_test, y_test, name='Best estimator\n', ax=ax1)
    roc_from_predictions = RocCurveDisplay.from_predictions(y_test, y_pred, name='From predictions\n', ax=ax1)
    ax1.plot([0, 1], ls="--", color='k', label='Base rate\n(AUC = 0.5)')
    ax1.set_title('Test set area under the ROC curve', fontsize=15)
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0))
    sns.despine()

    for i in range(len(precs)):
        model_name = model.best_estimator_.steps[-1][1].__class__.__name__
        label_model = f"{model_name}\nAUC = {pr_aucs[i]:.3f}"
        ax2.plot(recs[i], precs[i], label=label_model)
    ax2.plot([0, 1], [0, 0], linestyle='--', label='Base rate', color='black')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Test set area under the PR curve', fontsize=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, frameon=True)
    plt.subplots_adjust(hspace=0.3)
    sns.despine()


print('Figure 7: Best model ROC and PR curves')
roc_pr_auc_curves_from_best_estimator(grid_lr)

# Predict target variable using the best estimator
y_pred_grid_lr = grid_lr.predict(X_test)

# Create a Confusion Matrix Display object
cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_grid_lr),
                            display_labels=['Not readmitted', 'Readmitted'])

print('Figure 8: GridSearchCV best estimator (Logistic Regression) confusion matrix.')
cm.plot(cmap='Blues')
plt.show()

# Get coefficients of the best Logistic Regression estimator
print('Figure 9: GridSearchCV best estimator (Logistic Regression) coefficient importances.')
coeffs = grid_lr.best_estimator_.named_steps.logisticregression.coef_[0]

# Sort coefficients
importances_model = pd.Series(coeffs[:len(X.columns)],
                              index=X
                              .columns[:len(coeffs[:len(X.columns)])]).sort_values()

# Plot  LogisticRegression coefficients
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

importances_model.plot(kind='barh', color=palette(importances_model / float(importances_model.max())))
plt.xlabel('Features', fontsize=12)
plt.ylabel('Feature importances', fontsize=12)

plt.show()