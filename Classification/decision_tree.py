# Decision Trees
print("------------DECISION TREES--------------------")
X = df_encoded_unscaled.copy()
y = df_predictive_unscaled.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

dt_clf = DecisionTreeClassifier(random_state=43)
ad.custom_crossvalidation(X_train, y_train, dt_clf)
dt_clf = dt_clf.fit(X_train, y_train)
plts.custom_plot_tree(dt_clf, X_train)
print(f"Tree Depth: {dt_clf.get_depth()} \nNumber of Leaves: {dt_clf.get_n_leaves()}")
print(f"Classes: {dt_clf.classes_}")
# print(f"Decision Path: {dt_clf.decision_path(X_train, check_input=True)}")

# Takes approx. 1 hour to run!
# params = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': range(3, 11),
#     'min_impurity_decrease': np.arange(0.01, 0.3, 0.01),
#     'min_samples_leaf': range(2, 20, 4),
#     'min_samples_split': range(2, 10, 2),
#     'ccp_alpha': [0.003, 0.005]
# }
# grid_search = GridSearchCV(dt_clf, param_grid=params, scoring='f1_macro')
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# Test model on testing portion
dt_clf = DecisionTreeClassifier(random_state=43, ccp_alpha=0.003, criterion='entropy', max_depth=5,
                                min_impurity_decrease=0.01, min_samples_leaf=2, min_samples_split=2)
ad.custom_crossvalidation(X_train, y_train, dt_clf)
dt_clf = dt_clf.fit(X_train, y_train)

# No scaling required for Decision Trees
cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', encoder, cat_columns)  # encode categorical columns
])

# pipeline = Pipeline([
#     ('preprocessing', preprocessor),
#     ('sklearn_dt', grid_search.best_estimator_)
# ])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', dt_clf)
])

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# print(f"Classification Report\n: {classification_report(y_test, y_pred)}")
# ad.plot_confusion_matrix(y_test, y_pred)
# plts.custom_plot_tree(dt_clf, X_test)
# print(f"Tree Depth: {dt_clf.get_depth()} \nNumber of Leaves: {dt_clf.get_n_leaves()}")
# print(f"Classes: {dt_clf.classes_}")

# Scaled - decision tress do not rely on numerical scale of the features
# they split data based on feature thresholds, these threshold dpeend only on the relative order of values, not
# their magnitude -  so in summary, Scaling for Decision Trees is not relevant
# X = df_adult_encoded_scaled.copy()
# y = df_adult_predictive.copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
# ad.custom_crossvalidation(X_train, y_train, dt_clf)
# dt_clf = dt_clf.fit(X_train, y_train)