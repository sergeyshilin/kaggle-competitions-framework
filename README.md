# kaggle-competitions-framework
Framework for fast prototyping and training of single models and ensembles.
Training happens according to a data generator you pass in, but initially designed to take a K-Fold split in order to produce both OOF and test predictions.

Below you can find a short API explanation, however it does not cover all the needs 

## Data Loader
### Initialization
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
### Initialization

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


To run a training of a single model it is only required to specify 3 parameters:
- path to data folder (where to get `train.csv` and `test.csv` from)
- path to models folder (where to save code sources to be able to reproduce your predictions later)
- path to predictions folder (for future stacking/blending)

```bash
python src/single_model.py --data 'data/' --models 'models/' --preds 'predictions/'
```
