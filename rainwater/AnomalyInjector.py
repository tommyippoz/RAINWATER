import copy
import random

from .TimeSeriesSequence import TimeSeriesSequence


def get_injectors(desc_list: list, duration: int, min_normal: int, perc_inj: float):
    """
    Returns a list of injectors
    :param desc_list: list of tags read from config file
    :param perc_inj: likelihood of activating an injection
    :param min_normal: minimum number of normal traces before injection
    :param duration: duration of the injection
    :return:
    """
    injectors = []
    if desc_list is not None:
        for item in desc_list:
            if item in ['increase_load', 'more_load', 'more', 'increase']:
                injectors.append(LoadMultiplier(duration, min_normal, perc_inj, 2))
            elif item in ['max_load', 'max']:
                injectors.append(LoadMultiplier(duration, min_normal, perc_inj, 3))
            elif item in ['min_load', 'min']:
                injectors.append(LoadMultiplier(duration, min_normal, perc_inj, 0.3))
            elif item in ['decrease_load', 'less_load', 'less', 'decrease']:
                injectors.append(LoadMultiplier(duration, min_normal, perc_inj, 0.5))
            elif item in ['block', 'stop', 'zero', 'null']:
                injectors.append(ZeroInjector(duration, min_normal, perc_inj))
            elif item in ['repeat', 'more', 'replay']:
                injectors.append(RepeatInjector(duration, min_normal, perc_inj))

    return injectors


class AnomalyInjector:
    """
    Used for injecting Anomalies in a Sequence. Provides a simple interfacing with an inject method.
    """

    def __init__(self, duration: int = 1, tag: str = 'injection',
                 min_normal: int = 5, perc_inj: float = 0.1):
        """
        Constructor
        :param duration: the duration of the injection (number of data points)
        :param tag: the tag associated to the injection
        :param min_normal: the amount of data points at the beginning of the sequence that have to be left unaltered
        :param perc_inj: the probability of the injection to happen at a given instant
        """
        self.duration = duration
        self.min_normal = min_normal
        self.perc_inj = perc_inj
        self.tag = tag + "@" + str(self.duration) + "@" + str(self.min_normal)

    def inject(self, seq: TimeSeriesSequence):
        """
        Method for injecting an anomaly inside a sequence
        :param seq: the sequence to be modified
        :return: the injected sequence
        """
        if seq is None:
            print('sequence to inject is None')
            return None
        if self.duration + self.min_normal > seq.length():
            print('[%s] sequence is too short %d/%d' % (self.tag, seq.length(), self.duration + self.min_normal))
            return None
        from_in = self.min_normal
        while (from_in < seq.length() - self.duration) and \
                (random.randint(0, 999) < self.perc_inj * 1000):
            from_in += 1
        to_in = min(from_in + self.duration, seq.length())
        inj_values = self.inject_anomaly(seq, from_in, to_in)
        new_values = copy.deepcopy(seq.get_values())
        new_values[from_in:to_in] = inj_values
        labels = ['normal' for _ in range(0, seq.length())]
        labels[from_in:to_in] = [self.tag for _ in range(0, to_in-from_in)]
        new_seq = TimeSeriesSequence(tag=seq.get_tag() + "-" + self.tag,
                                     times=seq.get_times(), values=new_values, labels=labels)
        return new_seq

    def get_name(self):
        """
        Gets the tag of the injection
        :return: a string (injection tag)
        """
        return self.tag

    def inject_anomaly(self, seq: TimeSeriesSequence, from_index: int, to_index: int):
        """
        Placeholder for the abstract inject_anomaly function to be overridden by child classes
        :param seq: the sequence to be injected
        :param from_index: start of the injection
        :param to_index: end of the injection
        :return: the injected sequence
        """
        return None


class LoadMultiplier(AnomalyInjector):

    def __init__(self, duration: int = 1, min_normal: int = 5,
                 perc_inj: float = 0.1, mult_factor: float = 2):
        """
        Constructor
        :param duration: the duration of the injection (number of data points)
        :param min_normal: the amount of data points at the beginning of the sequence that have to be left unaltered
        :param perc_inj: the probability of the injection to happen at a given instant
        :param mult_factor: the float value used as multiplier for the consumption
        """
        super().__init__(duration, 'load-multiplier-' + str(mult_factor), min_normal, perc_inj)
        self.mult_factor = mult_factor

    def inject_anomaly(self, seq: TimeSeriesSequence, from_index: int, to_index: int):
        """
        Implements the injection
        :param seq: the sequence to be injected
        :param from_index: start of the injection
        :param to_index: end of the injection
        :return: the injected sequence
        """
        series_slice = seq.get_values()[from_index:to_index]
        inj_array = series_slice*self.mult_factor
        return inj_array


class ZeroInjector(AnomalyInjector):

    def __init__(self, duration: int = 1, min_normal: int = 5, perc_inj: float = 0.1):
        """
        Constructor
        :param duration: the duration of the injection (number of data points)
        :param min_normal: the amount of data points at the beginning of the sequence that have to be left unaltered
        :param perc_inj: the probability of the injection to happen at a given instant
        """
        super().__init__(duration, 'zero', min_normal, perc_inj)

    def inject_anomaly(self, seq: TimeSeriesSequence, from_index: int, to_index: int):
        """
        Implements the injection
        :param seq: the sequence to be injected
        :param from_index: start of the injection
        :param to_index: end of the injection
        :return: the injected sequence
        """
        series_slice = seq.get_values()[from_index:to_index]
        inj_array = series_slice * 0.0
        return inj_array


class RepeatInjector(AnomalyInjector):

    def __init__(self, duration: int = 1, min_normal: int = 5, perc_inj: float = 0.1):
        """
        Constructor
        :param duration: the duration of the injection (number of data points)
        :param min_normal: the amount of data points at the beginning of the sequence that have to be left unaltered
        :param perc_inj: the probability of the injection to happen at a given instant
        """
        super().__init__(duration, 'repeat', min_normal, perc_inj)

    def inject_anomaly(self, seq: TimeSeriesSequence, from_index: int, to_index: int):
        """
        Implements the injection
        :param seq: the sequence to be injected
        :param from_index: start of the injection
        :param to_index: end of the injection
        :return: the injected sequence
        """
        series_slice = seq.get_values()[from_index:to_index]
        inj_array = seq.get_values()[from_index-1] + series_slice * 0.0
        return inj_array