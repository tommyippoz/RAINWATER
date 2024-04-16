import copy

import numpy
import pandas


def sequences_to_dataset(sequences: list, force_binary: bool = False):
    """
    Transforms a list of TimeSeriesSequence to dataset for ML
    :param sequences: list of sequences
    :return: the features and the label
    """
    data_list = []
    for sequence in sequences:
        data_list.append(sequence.get_all_data())
    dataset = pandas.concat(data_list, ignore_index=True)
    label = dataset['label'].to_numpy()
    if force_binary:
        label = numpy.where(label == 'normal', 'normal', 'anomaly')
    dataset = dataset.drop(columns=['timestamp', 'label'])
    return dataset, label


class TimeSeriesSequence:

    def __init__(self, tag: str, times: numpy.ndarray, values: numpy.ndarray, labels: numpy.ndarray = None):
        self.tag = tag
        self.times = times
        self.values = values
        self.labels = labels
        self.additional_features = None
        if self.labels is None:
            self.labels = ['normal' for _ in range(0, len(values))]

    def get_times(self):
        return self.times

    def get_values(self):
        return self.values

    def get_labels(self):
        return self.labels

    def get_i(self, index: int):
        return [self.times[index], self.values[index], self.labels[index]]

    def length(self):
        if self.values is not None:
            return len(self.values)
        else:
            return 0

    def get_tag(self):
        return self.tag

    def add_features(self):
        # Init DataFrame
        new_f = pandas.DataFrame()
        new_f['base'] = self.values
        new_f['time'] = self.times

        # Init vars
        min_diff = int((self.times[1] - self.times[0]).astype('timedelta64[m]') / numpy.timedelta64(1, 'm'))
        obs_next_hour = int(60 / min_diff)
        obs_next_day = 24 * obs_next_hour

        # General Features
        new_f['dayofweek'] = new_f['time'].dt.dayofweek
        new_f['is_weekday'] = (new_f['dayofweek'] < 5) * 1
        new_f['is_weekend'] = (new_f['dayofweek'] >= 5) * 1

        # Differences Between Features
        new_f['diff t-1'] = new_f['base'] - new_f['base'].shift(1)
        new_f['diff t-2'] = new_f['base'] - new_f['base'].shift(2)
        new_f['diff t-3'] = new_f['base'] - new_f['base'].shift(3)
        new_f['diff t-5'] = new_f['base'] - new_f['base'].shift(5)
        new_f['diff t-10'] = new_f['base'] - new_f['base'].shift(10)
        new_f['diff hour'] = new_f['base'] - new_f['base'].shift(obs_next_hour)
        new_f['diff day'] = new_f['base'] - new_f['base'].shift(obs_next_day)
        new_f = new_f.fillna(0)

        #  Relative Differences between Features
        new_f['rdiff t-1'] = [new_f['diff t-1'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                              for i in new_f.index]
        new_f['rdiff t-2'] = [new_f['diff t-2'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                              for i in new_f.index]
        new_f['rdiff t-3'] = [new_f['diff t-3'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                              for i in new_f.index]
        new_f['rdiff t-5'] = [new_f['diff t-5'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                              for i in new_f.index]
        new_f['rdiff t-10'] = [new_f['diff t-10'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                               for i in new_f.index]
        new_f['rdiff t-hour'] = [new_f['diff hour'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                                 for i in new_f.index]
        new_f['rdiff t-day'] = [new_f['diff day'][i] / new_f['base'][i] if new_f['base'][i] != 0 else 1
                                for i in new_f.index]
        new_f = new_f.fillna(1)

        # Moving Averages
        new_f['diff ma-2'] = new_f['base'] - new_f.rolling(window=2)['base'].mean()
        new_f['diff ma-3'] = new_f['base'] - new_f.rolling(window=3)['base'].mean()
        new_f['diff ma-5'] = new_f['base'] - new_f.rolling(window=5)['base'].mean()
        new_f['diff ma-10'] = new_f['base'] - new_f.rolling(window=10)['base'].mean()
        new_f['diff ma-hour'] = new_f['base'] - new_f.rolling(window=obs_next_hour)['base'].mean()
        new_f['diff ma-day'] = new_f['base'] - new_f.rolling(window=obs_next_day)['base'].mean()
        new_f = new_f.fillna(0)

        self.additional_features = new_f.drop(columns=['base', 'time'])

    def get_all_data(self):
        all_df = copy.deepcopy(self.additional_features)
        all_df['timestamp'] = self.times
        all_df['consumption'] = self.values
        all_df['label'] = self.labels
        return all_df
