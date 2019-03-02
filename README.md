# kaggle-competitions-framework
Framework for typical Kaggle ensembling

To run a training of a single model it is only required to specify 3 parameters:
- path to data folder (where to get `train.csv` and `test.csv` from)
- path to models folder (where to save code sources to be able to reproduce your predictions later)
- path to predictions folder (for future stacking/blending)

```bash
python src/single_model.py --data 'data/' --models 'models/' --preds 'predictions/'
```
