import pandas as pd
from numpy.random import seed
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

seed(0)


def create_pipeline(model):
    return make_pipeline(SimpleImputer(strategy='median'), model)


def test_model(model, name):
    if name == GRADIENT_BOOSTING:
        model.fit(X_train, y_train.values,
                  xgbclassifier__early_stopping_rounds=GRADIENT_BOOSTING_STOP,
                  xgbclassifier__eval_set=[(X_valid, y_valid)],
                  xgbclassifier__verbose=False)
    else:
        model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)
    print(f'{name} performance: {accuracy}')
    return predictions


def voting_prediction(predictions):
    voted_predictions = list()
    for i in range(len(predictions[0])):
        row_predictions = [p[i] for p in predictions]
        vote = max(row_predictions, key=row_predictions.count)
        voted_predictions.append(vote)
    return voted_predictions


TARGET_FEATURE = 'severity'
ALL_FEATURES = ['BI-RADS', 'age', 'shape', 'margin', 'density',
                TARGET_FEATURE]

RANDOM_FOREST_ESTIMATORS = 100
EXTRA_TREES_ESTIMATORS = 100
GRADIENT_BOOSTING_ESTIMATORS = 500
GRADIENT_BOOSTING_STOP = 15
GRADIENT_BOOSTING_LEARNING_RATE = 0.01
GRADIENT_BOOSTING = 'Gradient Boosting'

X = pd.read_csv('mammographic_masses.data.txt',
                na_values=['?'], names=ALL_FEATURES)
X.dropna(axis=0, subset=[TARGET_FEATURE])
X['shape_density'] = X['shape'] * X['density']
print('Correlations:')
print(X.corr())
print()

y = X[TARGET_FEATURE]
X.drop([TARGET_FEATURE], inplace=True, axis=1)

print('Selected features:')
SELECTED_FEATURES = ['BI-RADS','shape', 'margin', 'density']
print(SELECTED_FEATURES)
X = X[SELECTED_FEATURES]
print()

X_train, X_valid, y_train, y_valid = train_test_split(X.values, y, train_size=0.75)
model = create_pipeline(DummyClassifier(strategy='most_frequent'))
test_model(model, 'Baseline')

model = create_pipeline(RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS))
random_forest_predictions = test_model(model, 'Random Forest')

model = create_pipeline(ExtraTreesClassifier(n_estimators=EXTRA_TREES_ESTIMATORS))
extra_trees_predictions = test_model(model, 'Extra trees')

model = create_pipeline(LogisticRegression(solver='lbfgs'))
logistic_regression_predictions = test_model(model, 'Logistic Regression')

model = create_pipeline(KNeighborsClassifier(n_neighbors=50))
k_nearest_predictions = test_model(model, 'K nearest neighbors')

model = create_pipeline(SVC(kernel='poly', C=1.0, gamma='scale'))
svc_predictions = test_model(model, 'SVC')

model = create_pipeline(XGBClassifier(n_estimators=GRADIENT_BOOSTING_ESTIMATORS,
                                      learning_rate=GRADIENT_BOOSTING_LEARNING_RATE))
gradient_boosting_predictions = test_model(model, GRADIENT_BOOSTING)

predictions = [random_forest_predictions, svc_predictions, logistic_regression_predictions,
               gradient_boosting_predictions]
voted_predictions = voting_prediction(predictions)
accuracy = accuracy_score(y_valid, voted_predictions)
print()
print(f'Voted(Random Forest, Svc, Logistic Regression, Gradient Boosting) accuracy')
print(accuracy)
