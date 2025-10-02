import configparser
import math
import os
import time

import numpy
import pandas
import sklearn.metrics

from rainwater import TimeSeriesSequence
from rainwater.DecisionPolicy import DecisionPolicy, policy_from_string


def get_classifier_name(clf_obj) -> str:
    """
    Gets the name of a classifier object
    :param clf_obj:
    :return:
    """
    invert_op = getattr(clf_obj, "classifier_name", None)
    if callable(invert_op):
        return clf_obj.classifier_name()
    else:
        return clf_obj.__class__.__name__

def read_csv(filepath: str, kwh_col: str = 'kWh', time_col: str = 'timestamp',
             seq_col: str = 'device_id', batch_size: int = None, limit_rows: int = -1):
    """
    Reads a dataset stored as csv file
    :param filepath: path to the file
    :param kwh_col: name of the csv column containing the consumption
    :param time_col: name of the csv column containing the timestamp
    :param seq_col: name of the csv column that is used to distinguish between series (if any)
    :param batch_size: (used in case seq_col does not exist) length of series in number of data points
    :param limit_rows: (optional) used to use a subset of the dataset
    :return: a list of time series
    """
    # Read the dataset
    dataset_name = filepath.split('/')[-1].replace(".csv", "")
    if limit_rows > 0:
        data = pandas.read_csv(filepath, sep=',', nrows=limit_rows)
    else:
        data = pandas.read_csv(filepath, sep=',')
    if kwh_col is None or kwh_col not in data.columns:
        print('kwh column \'%s\' does not exist' % kwh_col)
        return None
    if time_col is None or time_col not in data.columns:
        print('timestamp column \'%s\' does not exist' % time_col)
        return None
    if (seq_col is None or seq_col not in data.columns) and batch_size is None:
        print('sequence column \'%s\' does not exist, no batch size specified' % seq_col)
        return None

    # Trim dataset
    data[time_col] = pandas.to_datetime(data[time_col])
    if seq_col not in data.columns:
        # Case in which only batch_size is specified
        seq_tags = numpy.concatenate([['batch' + str(i + 1) for _ in range(0, batch_size)]
                                      for i in range(0, int(len(data.index) / batch_size) + 1)],
                                     axis=None)
        data['seq_col'] = seq_tags[0:len(data.index)]
        seq_col = 'seq_col'
    elif seq_col in data.columns and batch_size > 0:
        # Case in which both batch_size and seq_col are specified
        change_ids = [v for v in data[(data[seq_col].shift() != data[seq_col]) != 0].index] + [len(data.index)]
        for i in range(0, len(change_ids) - 1):
            device_length = change_ids[i + 1] - change_ids[i]
            seq_tags = numpy.concatenate([[str(data[seq_col][change_ids[i]]) + '_' + str(k + 1)
                                           for _ in range(0, batch_size)]
                                          for k in range(0, int(device_length / batch_size) + 1)],
                                         axis=None)
            data[seq_col][change_ids[i]:change_ids[i + 1]] = seq_tags[0:device_length]
    data = data[[kwh_col, time_col, seq_col]]
    split_idx = [v for v in data[(data[seq_col].shift() != data[seq_col]) != 0].index if v > 0]

    # Create Sequences
    sequences = []
    for seq_data in numpy.split(data, split_idx):
        seq_data.reset_index(inplace=True)
        seq_tag = dataset_name + '_' + str(seq_data[seq_col][0])
        sequences.append(TimeSeriesSequence(seq_tag, numpy.asarray(seq_data[time_col]),
                                            numpy.asarray(seq_data[kwh_col], dtype=float), None))

    return sequences


def load_dataset_config(cfg_file):
    """
    Loads parameters for execution from a configuration file. Specific for dataset loaders
    :param cfg_file: the path to the configuration file
    :return: a dictionary
    """
    # Loading Conf file
    if not os.path.exists(cfg_file):
        return None
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # Setting up variables
    cfg_info = {}
    if config is not None:
        # Reading CSV info
        if ('csv_path' in config['base']) and len(config['base']['csv_path'].strip()) > 0 \
                and os.path.exists(config['base']['csv_path'].strip()):
            cfg_info['csv_path'] = config['base']['csv_path'].strip()
        else:
            cfg_info['csv_path'] = None
        if ('kwh_col' in config['base']) and len(config['base']['kwh_col'].strip()) > 0:
            cfg_info['kwh_col'] = config['base']['kwh_col'].strip()
        else:
            cfg_info['kwh_col'] = None
        if ('time_col' in config['base']) and len(config['base']['time_col'].strip()) > 0:
            cfg_info['time_col'] = config['base']['time_col'].strip()
        else:
            cfg_info['time_col'] = None
        if ('seq_col' in config['base']) and len(config['base']['seq_col'].strip()) > 0:
            cfg_info['seq_col'] = config['base']['seq_col'].strip()
        else:
            cfg_info['seq_col'] = None
        if ('batch_size' in config['base']) and len(config['base']['batch_size'].strip()) > 0:
            cfg_info['batch_size'] = int(config['base']['batch_size'].strip())
        else:
            cfg_info['batch_size'] = -1
        if ('limit_rows' in config['base']) and len(config['base']['limit_rows'].strip()) > 0:
            cfg_info['limit_rows'] = int(config['base']['limit_rows'].strip())
        else:
            cfg_info['limit_rows'] = -1

            # Threats
        cfg_info['threats'] = []
        for thr, flag in config.items("threats"):
            if flag:
                cfg_info['threats'].append(thr)

        # Injection
        if ('duration' in config['injection']) and len(config['injection']['duration'].strip()) > 0:
            cfg_info['duration'] = int(config['injection']['duration'].strip())
        if ('min_normal' in config['injection']) and len(config['injection']['min_normal'].strip()) > 0:
            cfg_info['min_normal'] = int(config['injection']['min_normal'].strip())
        if ('perc_inj' in config['injection']) and len(config['injection']['perc_inj'].strip()) > 0:
            cfg_info['perc_inj'] = float(config['injection']['perc_inj'].strip())

    return cfg_info


def read_general_conf(cfg_file):
    """
    Loads parameters for execution from a configuration file.
    :param cfg_file: the path to the configuration file
    :return: a dictionary
    """
    # Loading Conf file
    if not os.path.exists(cfg_file):
        return None
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # Setting up variables
    cfg_info = {}
    if config is not None:
        # Reading folders info
        if ('cfg_files_folder' in config['folders']) and len(config['folders']['cfg_files_folder'].strip()) > 0:
            cfg_info['input_folder'] = config['folders']['cfg_files_folder'].strip()
            if not os.path.exists(cfg_info['input_folder']):
                os.mkdir(cfg_info['input_folder'])
        else:
            cfg_info['input_folder'] = None
        if ('models_folder' in config['folders']) and len(config['folders']['models_folder'].strip()) > 0:
            cfg_info['models_folder'] = config['folders']['models_folder'].strip()
            if not os.path.exists(cfg_info['models_folder']):
                os.mkdir(cfg_info['models_folder'])
        else:
            cfg_info['models_folder'] = None
        if ('out_folder' in config['folders']) and len(config['folders']['out_folder'].strip()) > 0:
            cfg_info['output_folder'] = config['folders']['out_folder'].strip()
            if not os.path.exists(cfg_info['output_folder']):
                os.mkdir(cfg_info['output_folder'])
        else:
            cfg_info['output_folder'] = None
        if ('gen_data_tag' in config['folders']) and len(config['folders']['gen_data_tag'].strip()) > 0:
            cfg_info['data_tag'] = config['folders']['gen_data_tag'].strip()
        else:
            cfg_info['data_tag'] = 'generated'

        # Reading trainer info
        if ('tt_split' in config['trainer']) and len(config['trainer']['tt_split'].strip()) > 0:
            cfg_info['tt_split'] = float(config['trainer']['tt_split'].strip())
        else:
            cfg_info['tt_split'] = 0.5
        if ('policy' in config['trainer']) and len(config['trainer']['policy'].strip()) > 0:
            policy_string = config['trainer']['policy'].strip()
            if ',' in policy_string:
                cfg_info['policy'] = [policy_from_string(ps.strip()) for ps in policy_string.split(',')]
            else:
                cfg_info['policy'] = [policy_from_string(policy_string)]
        else:
            cfg_info['policy'] = DecisionPolicy.NONE
        if ('force_binary' in config['trainer']) and len(config['trainer']['force_binary'].strip()) > 0:
            fb_text = config['trainer']['force_binary'].strip()
            cfg_info['force_binary'] = True if fb_text in ['1', 'True', 'true', 'yes', 'Y', 'T'] else False
        else:
            cfg_info['force_binary'] = False

    return cfg_info


def print_sequences(filename: str, seq_list, create_new=True):
    """
    Prints a list of sequences to a file
    :param filename: name of the file
    :param seq_list: list of sequences to print
    :param create_new: True if the files needs to be created, otherwise it appends
    """

    open_flag = 'w' if create_new else 'a'
    with open(filename, open_flag) as f:
        if open_flag == 'w':
            f.write('seq_name,timestamp,value,label\n')
        for seq in seq_list:
            if seq is not None:
                for i in range(0, seq.length()):
                    element = seq.get_i(i)
                    f.write("%s,%s,%f,%s\n" % (seq.get_tag(), element[0], element[1], element[2]))


def print_full_sequences(filename: str, seq_list, create_new=True):
    """
    Prints a list of sequences to a file
    :param filename: name of the file
    :param seq_list: list of sequences to print
    :param create_new: True if the files needs to be created, otherwise it appends
    """

    open_flag = 'w' if create_new else 'a'
    for seq in seq_list:
        if seq is not None:
            to_write = seq.get_all_data()
            to_write['seq_name'] = [seq.tag for _ in to_write.index]
            to_write.to_csv(filename, mode=open_flag, index=False, header=(False if open_flag == 'a' else True))
            open_flag = 'a'


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def compute_stats(clf_y, ad_y, test_y, test_seq: list = None, diagnosis_time: int = 1):
    """
    Computes stats for both the classifier and the anomaly detector
    :param clf_y: scores of the classifier
    :param ad_y: scores of the detector (including policy)
    :param test_y: ground truth
    :return: a dictionary
    """
    stats = {}
    classes = numpy.unique(test_y)
    stats['clf'] = compute_single_stats(clf_y, test_y, classes)
    stats['ad'] = compute_single_stats(ad_y, test_y, classes, test_seq, diagnosis_time)
    stats['classes'] = "-".join([x for x in classes])
    return stats


def compute_single_stats(pred_y, test_y, classes, test_sequences=None,
                         diagnosis_time: int = 1, max_delay: int = 50):
    """
    Computes stats for predicted labels
    :param pred_y: scores of the classifier
    :param test_y: ground truth
    :param classes: problem's classes
    :return: a dictionary
    """
    stats = {}
    stats['accuracy'] = sklearn.metrics.balanced_accuracy_score(test_y, pred_y)
    stats['mcc'] = sklearn.metrics.matthews_corrcoef(test_y, pred_y)
    stats['recall_class'] = dict(zip(classes,
                                     sklearn.metrics.recall_score(test_y, pred_y,
                                                                  labels=classes, average=None)))
    if test_sequences is not None:
        normal_class = 'normal'
        fprs = []
        dds = []
        dds_per_class = {x: [] for x in classes}
        tprs = []
        tprs_per_class = {x: [] for x in classes}
        index = 0
        for seq in test_sequences:

            # Deriving sequence-related scores
            seq_pred = pred_y[index:index + seq.length()]
            seq_label = test_y[index:index + seq.length()]
            index += seq.length()

            # Iterating over sequence data
            cooldown = 0
            det_label = ''
            an_time = -1
            det_time = -1
            false_alerts = []
            for i in range(0, len(seq_pred)):
                if cooldown == 0 and seq_pred[i] != normal_class:
                    cooldown = diagnosis_time
                    if det_time < 0 and seq_label[i] == seq_pred[i]:
                        det_time = i
                    if seq_label[i] != seq_pred[i]:
                        false_alerts.append(i)
                if an_time < 0 and seq_label[i] != normal_class:
                    an_time = i
                    det_label = seq_label[i]
                if cooldown > 0:
                    cooldown -= 1

            # Updating metrics
            fprs.append(len(false_alerts) / (seq_label == normal_class).sum())
            detected = 1 if det_time > 0 and not math.isnan(det_time) else 0
            tprs.append(detected)
            tprs_per_class[det_label].append(detected)
            if detected > 0:
                dds.append((det_time - an_time) if an_time <= det_time and not math.isnan(det_time) else max_delay)
                dds_per_class[det_label].append(
                    (det_time - an_time) if an_time <= det_time and not math.isnan(det_time) else max_delay)

        stats['overall_fpr'] = {'avg': float(numpy.average(fprs)), 'min': float(numpy.min(fprs)),
                                'max': float(numpy.max(fprs)), 'median': float(numpy.median(fprs)),  # 'all': fprs
                                }
        stats['overall_tpr'] = {'avg': float(numpy.average(tprs)), 'min': int(numpy.min(tprs)),
                                'max': int(numpy.max(tprs)), 'median': int(numpy.median(tprs)),  # 'all': tprs
                                }
        stats['overall_dd'] = {'avg': float(numpy.nanmean(dds)), 'min': int(numpy.nanmin(dds)),
                               'max': int(numpy.nanmax(dds)), 'median': int(numpy.nanmedian(dds)),  # 'all': dds
                               } if len(dds) > 0 else {'avg': -1, 'min': -1, 'max': -1, 'median': -1}
        stats['per_class_tpr'] = {}
        stats['per_class_dd'] = {}
        for an_class in tprs_per_class.keys():
            class_tpr = tprs_per_class[an_class]
            stats['per_class_tpr'][an_class] = {'avg': float(numpy.average(class_tpr)) if len(class_tpr) > 0 else -1,
                                                'min': int(numpy.min(class_tpr)) if len(class_tpr) > 0 else -1,
                                                'max': int(numpy.max(class_tpr)) if len(class_tpr) > 0 else -1,
                                                'median': int(numpy.median(class_tpr)) if len(class_tpr) > 0 else -1,  # 'all': tprs
                                                }
            class_dd = dds_per_class[an_class]
            stats['per_class_dd'][an_class] = {'avg': float(numpy.nanmean(class_dd)) if len(class_dd) > 0 else -1,
                                               'min': int(numpy.nanmin(class_dd)) if len(class_dd) > 0 else -1,
                                               'max': int(numpy.nanmax(class_dd)) if len(class_dd) > 0 else -1,
                                               'median': int(numpy.nanmedian(class_dd)) if len(class_dd) > 0 else -1,  # 'all': dds
                                               } if len(class_dd) > 0 else {'avg': -1, 'min': -1, 'max': -1, 'median': -1}

    return stats
