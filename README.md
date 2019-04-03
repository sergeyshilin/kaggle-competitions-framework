# kaggle-competitions-framework
Framework for fast prototyping and training of single models and ensembles.
Training happens according to a data generator you pass in, but initially designed to take a K-Fold split in order to produce both OOF and test predictions.

Below you can find a short API explanation, however it does not cover all the possible use cases. For more information, please check out [examples](examples/).

## Table of Contents
- [Data Loader](#data-loader)
  - [Initialization](#data-loader-initialization)
  - [Data Preprocessors](#data-preprocessors)
  - [Data Generator](#data-generator)
- [Model Loader](#model-loader)
  - [Initialization](#model-loader-initialization)
  - [Initialize a custom model](#initialize-a-custom-model)
  - [Run training](#run-training)
  - [Save results](#save-results)
- [Contribution](#contribution)


## Data Loader
### Data Loader Initialization
```python
from data import DataLoader

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(data_folder, **dl_params)
```
`data_folder` is a path to a folder where `train.csv` and `test.csv` files can be found

`dl_parameters` specifies a target and ID columns in your `train.csv` file. Target will be removed from the data before the training starts.

### Data Preprocessors
```python
from data.preprocessors import GenericDataPreprocessor
from sklearn.preprocessing import StandardScaler

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X.drop(['ID_code'], axis=1)

data_loader.preprocess(DropColumns, StandardScaler, ToNumpy)
```

It will also work with custom arguments.

```python
from sklearn.preprocessing import MinMaxScaler
data_loader.preprocess(MinMaxScaler, feature_range=(-1, 1))
```

All preprocessors must have `fit_transform` and `transform` functions implemented, thus sklearn transformers can be applied to the `DataLoader`. Custom preprocessors must inherit `GenericDataPreprocessor` base class.

### Data Generator
```python
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
```

## Model Loader
### Model Loader Initialization

Models are initialized with three main arguments, the model class, loader parameters and the model parameters.
```python
model_params = {
    'name':     "dense_nn",
    'fit':      "fit",
    'predict':  "predict_proba",
    'pred_col': 1
}

nn_params = {
    'build_fn': dense_nn_model,
    'epochs': 25,
    'batch_size': 256,
    'verbose': 1
}

model = ModelLoader(KerasClassifier, loader_params, **nn_params)
```

### Initialize a custom model

If the model does not have a sklearn-like interface, it is still possible to create a custom model interface, inherited from `GenericModel` base class. `fit` and `predict` function must be implemented as following

```python
class LightGbmTrainer(GenericModel):
    def __init__(self):
        self.lgb_params = {
            "objective" : "binary",
            "metric" : "auc",
            "boosting": 'gbdt',
            "max_depth" : 4,
            "learning_rate" : 0.01,
            "bagging_fraction" : 0.8,
            "tree_learner": "serial",
            "verbosity" : 0,
        }

    def fit(self, train, cv):
        x_tr, y_tr = train
        x_cv, y_cv = cv
        trn_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_cv, label=y_cv)
        evals_result = {}
        self.model = lgb.train(self.lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result)

    def predict(self, test):
        return self.model.predict(test)


model_params = {
    'name':          "lightgbm",
    'fit':           "fit",
    'predict':       "predict"
}

model = ModelLoader(LightGbmTrainer, model_params)
```

### Run training
```python
fit_params = {
    'use_best_model': True,
    'verbose': 100,
    'plot': True
}
predict_params = {}

from sklearn.metrics import roc_auc_score
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)
```

### Save results

```python
current_file_path = os.path.abspath(__file__) # to save this .py file
model.save(data_loader, results, current_file_path, preds_folder, models_folder)
```

Where 

`models_folder` is a path where to save code sources to be able to reproduce your predictions later. In other words, it places your `current_file_path` into `models_folder`
`preds_folder` is a path to predictions folder (for future stacking/blending)

## Contribution

Feel free to send your pull request if would like anything to be improved.

We also use GitHub issues to track requests and bugs.
