import numpy
import pandas


class TimeSeriesSequence:

    def __init__(self, tag: str, times: numpy.ndarray, values: numpy.ndarray, labels: numpy.ndarray = None):
        self.tag = tag
        if labels is None:
            labels = ['normal' for _ in range(0, len(values))]
        self.seq_data = pandas.DataFrame()
        self.seq_data['times'] = times
        self.seq_data['values'] = values
        self.seq_data['labels'] = labels

    def get_times(self):
        return self.seq_data['times'].to_numpy()

    def get_values(self):
        return self.seq_data['values'].to_numpy()

    def get_labels(self):
        return self.seq_data['labels'].to_numpy()

    def get_i(self, index: int):
        return self.seq_data.iloc[index]

    def length(self):
        return len(self.seq_data.index)

    def get_tag(self):
        return self.tag

    def add_features(self):
        f_name = 'lumped_kWh'

        # differences
        batch['diff t-1'] = batch[f_name] - batch[f_name].shift(1)
        batch['diff t-2'] = batch[f_name] - batch[f_name].shift(2)
        batch['diff t-3'] = batch[f_name] - batch[f_name].shift(3)
        batch['diff t-4'] = batch[f_name] - batch[f_name].shift(4)
        batch['diff t-5'] = batch[f_name] - batch[f_name].shift(5)
        batch = batch.fillna(0)

        #  relative differences
        batch['rdiff t-1'] = [batch['diff t-1'][i] / batch[f_name][i] if batch[f_name][i] != 0 else 1
                              for i in batch.index]
        batch['rdiff t-2'] = [batch['diff t-2'][i] / batch[f_name][i] if batch[f_name][i] != 0 else 1
                              for i in batch.index]
        batch['rdiff t-3'] = [batch['diff t-3'][i] / batch[f_name][i] if batch[f_name][i] != 0 else 1
                              for i in batch.index]
        batch['rdiff t-4'] = [batch['diff t-4'][i] / batch[f_name][i] if batch[f_name][i] != 0 else 1
                              for i in batch.index]
        batch['rdiff t-5'] = [batch['diff t-5'][i] / batch[f_name][i] if batch[f_name][i] != 0 else 1
                              for i in batch.index]
        batch = batch.fillna(1)

        # moving averages
        batch['diff ma-2'] = batch[f_name] - batch.rolling(window=2)[f_name].mean()
        batch['diff ma-3'] = batch[f_name] - batch.rolling(window=3)[f_name].mean()
        batch['diff ma-4'] = batch[f_name] - batch.rolling(window=4)[f_name].mean()
        batch['diff ma-5'] = batch[f_name] - batch.rolling(window=5)[f_name].mean()
        batch = batch.fillna(0)

        # cleanup
        batch = batch.drop(columns=['index'])
        return batch