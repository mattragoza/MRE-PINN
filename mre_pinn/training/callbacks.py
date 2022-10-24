import deepxde


class PeriodicCallback(deepxde.callbacks.Callback):

    def __init__(self, period):
        super().__init__()
        self.period = period

    def on_batch_begin(self):
        if self.model.train_state.step % self.period != 0:
            return
        self.on_period_begin()

    def on_batch_end(self):
        if (self.model.train_state.step + 1) % self.period != 0:
            return
        self.on_period_end()

    def on_period_begin(self):
        pass

    def on_period_end(self):
        pass


class PDEResampler(PeriodicCallback):
    '''
    Resample PDE and BC training points each period.
    '''
    def on_period_end(self):
        self.model.data.train_x_all = None
        self.model.data.train_x_bc = None
        self.model.data.resample_train_points()


class SummaryDisplay(deepxde.display.TrainingDisplay):
    '''
    Training display that only prints a summary at
    the end of training instead of after every test.
    '''
    def print_one(self, *args, **kwargs):
        return
