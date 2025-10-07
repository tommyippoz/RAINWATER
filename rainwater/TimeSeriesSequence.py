import copy
import random

import numpy
import pandas


def sequences_to_dataset(sequences: list, force_binary: bool = False):
    """
    Transforms a list of TimeSeriesSequence to dataset for ML
    :param force_binary: True if you want classification to be forced as binary
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


def sequences_to_series_dataset(sequences: list, series_size: int = 3,
                                force_binary: bool = False, only_values: bool = False):
    """
    Transforms a list of TimeSeriesSequence to dataset for ML with LSTM
    :param series_size: the length of each sequence
    :param force_binary: True if you want classification to be forced as binary
    :param sequences: list of sequences
    :return: the features and the label as ndarrays
    """
    dataset = []
    label = []
    for sequence in sequences:
        if only_values:
            seq_data = sequence.get_base_data()
        else:
            seq_data = sequence.get_all_data()
        seq_label = seq_data['label'].to_numpy()
        seq_data = seq_data.drop(columns=['timestamp', 'label']).to_numpy()
        for i in range(0, len(seq_label)):
            if i + 1 < series_size:
                new_data = list(seq_data[0:i+1])
                for j in range (i, series_size-1):
                    new_data = [new_data[i]*(100-(j-i+1))/100] + new_data
                new_data = numpy.asarray(new_data)
            else:
                new_data = seq_data[i+1-series_size:i+1]
            dataset.append(new_data)
            new_label = seq_label[i]
            label.append(new_label)
    dataset = numpy.asarray(dataset)
    label = numpy.asarray(label)
    if force_binary:
        label = numpy.where(label == 'normal', 'normal', 'anomaly')
    return dataset, label


def sequences_to_unknown_dataset(sequences: list, train_flag: bool = True):
    """
    Transforms a list of TimeSeriesSequence to dataset for ML
    :param train_flag: True if this is training data
    :param sequences: list of sequences
    :return: the features and the label
    """
    out_data = {}
    tags = list(numpy.unique([seq.tag.split("#")[1] for seq in sequences]))
    tags.append("normal")
    for an_tag in tags:
        if an_tag == "normal":
            data_list = []
            for sequence in sequences:
                data_list.append(sequence.get_all_data())
            seqs = copy.deepcopy(sequences)
        elif train_flag:
            data_list = []
            seqs = []
            for sequence in sequences:
                if an_tag not in sequence.tag:
                    data_list.append(sequence.get_all_data())
                    seqs.append(sequence)
        else:
            data_list = []
            seqs = []
            for sequence in sequences:
                if an_tag in sequence.tag:
                    data_list.append(sequence.get_all_data())
                    seqs.append(sequence)
        dataset = pandas.concat(data_list, ignore_index=True)
        label = numpy.where(dataset['label'].to_numpy() == 'normal', 'normal', 'anomaly')
        dataset = dataset.drop(columns=['timestamp', 'label'])
        out_data[an_tag] = [dataset, label, seqs]
    return out_data


class TimeSeriesSequence:
    """
    Class that defines the object of a time-ordered sequence (of power consumption)
    """

    def __init__(self, tag: str, times: numpy.ndarray, values: numpy.ndarray, labels: numpy.ndarray = None):
        """
        Constructor
        :param tag: the tag associated to the sequence
        :param times: the timestamps
        :param values: the consumption values
        :param labels: the labels (if none, all considered normal)
        """
        self.tag = tag
        self.times = times
        self.values = values
        self.labels = labels
        self.additional_features = pandas.DataFrame()
        if self.labels is None:
            self.labels = ['normal' for _ in range(0, len(values))]

    def get_times(self):
        """
        Gets the timestamps of a sequence
        :return: a list
        """
        return self.times

    def get_values(self):
        """
        Gets the values of the sequence
        :return: a list
        """
        return self.values

    def get_labels(self):
        """
        Gets the labels of the sequence
        :return: a list
        """
        return self.labels

    def get_i(self, index: int):
        """
        Getter for the i-th item of the sequence
        :param index: the index of the item
        :return: a triple of [timestamp, consumption value (+ additional features), label]
        """
        return [self.times[index], self.values[index], self.labels[index]]

    def length(self):
        """
        Returns the length of the sequence
        :return: an int
        """
        if self.values is not None:
            return len(self.values)
        else:
            return 0

    def get_tag(self):
        """
        Returns the tag for the sequence
        :return: a string
        """
        return self.tag

    def add_features(self):
        """
        Creates additional features starting from the consumption feature
        :return: nothing
        """
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

    def get_base_data(self):
        all_df = pandas.DataFrame()
        all_df['timestamp'] = self.times
        all_df['consumption'] = self.values
        all_df['label'] = self.labels
        return all_df
