# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline


np.random.seed(1234)

# %%

def model_definition(model, X_train, y_train):
    model_name = str(model).split('(')[0]
    steps = [('model', model)]

    pipeline = Pipeline(steps=steps)
    pipeline.fit(X_train, y_train)

    return model_name, pipeline
# %%
'''
CREATION OF TRAINING AND TESTING SAMPLES
'''
dataset = pd.read_csv("data.csv", sep=';', 
            index_col=['pipe_id', 'material', 'inspection', 'type_pipe'])


x_cols = ['upstream_length', 'num_upstream_pipes',
       'pipe_length', 'pipe_age',
       'pipe_height', 'pipe_width', 'connection_surface', 'depth_avg', 'slope',
       'mat_Asbestos', 'mat_Beton', 'mat_BmL', 
       'mat_PE', 'mat_PEHD', 'mat_PP', 'mat_PRC',
       'mat_PVC', 'mat_PVCU', 'mat_Steinzeug',
       'type_KM', 'type_KR', 'Coord_X', 'Coord_Y']

standard_cols = ['upstream_length', 'num_upstream_pipes',
       'pipe_length', 'pipe_height', 'pipe_width', 'connection_surface', 'depth_avg', 'pipe_age',
       'slope', 'Coord_X', 'Coord_Y']

# Divide the dataset into train and test sets
train_set = dataset.sample(frac=.9, random_state=1234)

# Add synthetic samples in the extreme points to force new pipes to be healthy and old pipes to be defective
young_pipes = train_set.copy()
young_pipes['pipe_age']=0
young_pipes['target']=0

old_pipes = train_set.copy()
old_pipes['pipe_age']=100
old_pipes['target']=1
train_set = pd.concat([train_set, young_pipes, old_pipes], axis=0)

test_set = dataset.drop(train_set.index)
test_set.sort_values('pipe_age', inplace=True)


# %%
scaler = MinMaxScaler()

X_train = train_set[x_cols]
X_train[standard_cols] = scaler.fit_transform(X_train[standard_cols])
y_train = train_set['target']

# Creation and scaling of the test datasets
X_test = test_set[x_cols]
X_test[standard_cols] = scaler.transform(X_test[standard_cols])
y_test = test_set['target']
# %%
'''
CROSS VALIDATED SCORE
'''
models = [LogisticRegressionCV(solver='saga', penalty="elasticnet", l1_ratios=[.1], max_iter=100000),
            DecisionTreeClassifier(),
            SVC(probability=True),
            XGBClassifier(),
            MLPClassifier(hidden_layer_sizes=(100, 50), batch_size=16, 
                max_iter=500, early_stopping=True),
            RandomForestClassifier()]

#%%
cv_folds = 10

cv = ShuffleSplit(n_splits=cv_folds, test_size=0.3, random_state=0)

# Creation of the results arrays
test_accuracy = np.empty(shape=(cv_folds, len(models)))
test_recall = np.empty(shape=(cv_folds, len(models)))
test_precision = np.empty(shape=(cv_folds, len(models)))
test_f1 = np.empty(shape=(cv_folds, len(models)))
test_roc_auc = np.empty(shape=(cv_folds, len(models)))

model_names = []

# Iteration over each of the selected models and the splits
for mod_ind, clf in enumerate(models):
    model_names.append(str(clf).split('(')[0])
    for split, (train_index, test_index) in enumerate(cv.split(X_train)):
        # Split the training set in training and validation sets
        X_train_cv = X_train.iloc[train_index]
        y_train_cv = y_train.iloc[train_index]

        # Fit the model to the new training set, within the folds
        clf.fit(X_train_cv, y_train_cv)

        y_pred = clf.predict(X_test)

        # The test score is applied to the independent test set
        test_accuracy[split, mod_ind] = accuracy_score(y_test, y_pred)
        test_recall[split, mod_ind] = recall_score(y_test, y_pred)
        test_precision[split, mod_ind] = precision_score(y_test, y_pred)
        test_f1[split, mod_ind] = f1_score(y_test, y_pred)

        # Get the AUC score
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
        test_roc_auc[split, mod_ind] = auc(fpr, tpr)

#%%
# Convert the results of the metrics to a pandas dataframe and save them as a csv
test_accuracy_df = pd.DataFrame(test_accuracy, columns = model_names)
test_recall_df = pd.DataFrame(test_recall, columns = model_names)
test_precision_df = pd.DataFrame(test_precision, columns = model_names)
test_f1_df = pd.DataFrame(test_f1, columns = model_names)
test_roc_auc_df = pd.DataFrame(test_roc_auc, columns = model_names)

test_accuracy_df.to_csv("./Results/Scores/Cross_validation_accuracy.csv", sep=';')
test_recall_df.to_csv("./Results/Scores/Cross_validation_recall.csv", sep=';')
test_precision_df.to_csv("./Results/Scores/Cross_validation_precision.csv", sep=';')
test_f1_df.to_csv("./Results/Scores/Cross_validation_f1.csv", sep=';')
test_roc_auc_df.to_csv("./Results/Scores/Cross_validation_auc.csv", sep=';')

# %%
'''
SIMULATION STEP
'''
years = 100

for model in models:

    model_name, pipeline = model_definition(model, X_train, y_train)

    probabilities = np.empty(shape=(len(dataset), years))

    for pipe in range(len(dataset)):

        X_simulation = dataset.iloc[[pipe]][x_cols]
        X_simulation = pd.DataFrame(np.repeat(X_simulation.values, years, axis=0), columns=x_cols)
        X_simulation['pipe_age'] = np.arange(0, years)
        X_simulation[standard_cols] = scaler.transform(X_simulation[standard_cols])

        # Predict the probability of being defective of a single pipe
        probabilities[pipe, :] = model.predict_proba(X_simulation)[:, 1]

    model_probabilities = pd.DataFrame(probabilities, index=dataset.index)

    model_probabilities.to_csv(f'./Results/Simulations/Probabilities_{model_name}.csv', sep=';')

# %%
'''
MONOTONICITY
'''
# Get a metric of monotonicity of the simulations
monotonous_metric = {}
for model in models:
    model_name = str(model).split('(')[0]

    # Get the simulations of every model
    df_mono = pd.read_csv(f"./Results/Simulations/Probabilities_{model_name}.csv", sep=';', index_col=['pipe_id', 'material', 'inspection', 'type_pipe'])

    # Apply np.diff to get the differences between columns of every row
    monotonous = np.diff(df_mono.values, axis=1)

    # If the value is smaller than 0, it's not monotonous
    monotonous_count = np.sum(monotonous<0, axis=1)

    monotonous_metric[model_name] = monotonous_count

monotonous_df = pd.DataFrame(monotonous_metric)

monotonous_stats = monotonous_df.describe()

monotonous_df.to_csv(f"./Results/Monotonicity/Monotonicity_pipes.csv", sep=';')
monotonous_stats.to_csv(f"./Results/Monotonicity/Monotonicity_stats.csv", sep=';')

# %%

'''
SIMULATION OF SCENARIOS
'''
# Fitting of the best model according to the accuracy metrics and monotonicity test
model = LogisticRegressionCV(solver='saga', penalty="elasticnet", l1_ratios=[.1], max_iter=100000)
_, pipeline = model_definition(model, X_train, y_train)

# Years to perform simulation
years = 100

# Get only the first inspection of each pipe
dataset_index = list(dataset.index.names)
unique_pipes_dataset = dataset.reset_index().drop_duplicates('pipe_id', keep='first').set_index(dataset_index)

# Probability of being defective for each pipe according to the model
def_probs = np.empty(shape=(len(unique_pipes_dataset), years))

for pipe in range(len(unique_pipes_dataset)):

    X_simulation = unique_pipes_dataset.iloc[[pipe]][x_cols]
    X_simulation = pd.DataFrame(np.repeat(X_simulation.values, years, axis=0), columns=x_cols)
    X_simulation['pipe_age'] = np.arange(0, years)
    X_simulation[standard_cols] = scaler.transform(X_simulation[standard_cols])

    def_probs[pipe, :] = pipeline.predict_proba(X_simulation)[:, 1]

# %%

# Definition of different thresholds for the probability of failure
prob_A = np.argmax(def_probs>0.25, axis=1)
prob_B = np.argmax(def_probs>.5, axis=1)
prob_C = np.argmax(def_probs>.75, axis=1)

unique_pipes_dataset['Prob_a'] = prob_A
unique_pipes_dataset['Prob_b'] = prob_B
unique_pipes_dataset['Prob_c'] = prob_C

age_first_insp = dataset.reset_index().sort_values(['pipe_id', 'pipe_age']).drop_duplicates('pipe_id', keep='first')['pipe_age']

diff = age_first_insp.values-prob_B
perc_too_early = np.sum(diff<0)/len(diff)
perc_too_late = np.sum(diff>0)/len(diff)

print(perc_too_early)
print(perc_too_late)

# %%
