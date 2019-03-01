import sys


class ModelLoader:
    def __init__(self, model, model_params, *args, **kwargs):
        self.parameters = model_params
        self.model = model(*args, **kwargs)
        self._set_parameters()

    def _init_default_parameters(self):
        self.model_name = 'defaultmodel'
        self.fit_name = 'fit'
        self.predict_name = 'predict'
        self.preds_col_num = 0

    def _set_parameters(self):
        self._init_default_parameters()

        if 'name' in self.parameters:
            self.model_name = self.parameters['name']
        if 'fit' in self.parameters:
            self.fit_name = self.parameters['fit']
        if 'predict' in self.parameters:
            self.predict_name = self.parameters['predict']
        if 'pred_col' in self.parameters:
            self.preds_col_num = self.parameters['pred_col']

    def fit(self, *args, **kwargs):
        sys.stdout.write("Start training the model '{}'... \n".format(
            self.model_name))
        sys.stdout.flush()
        return getattr(self.model, self.fit_name)(*args, **kwargs)

    def get_parameter(self, parameter):
        return self.parameters[parameter]

    def predict(self, *args, **kwargs):
        return getattr(self.model, self.predict_name)(*args, **kwargs)

    def get_preds_col_number(self):
        return self.preds_col_num
