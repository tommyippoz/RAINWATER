import configparser
import os

import numpy
import pandas

from rainwater import TimeSeriesSequence


def read_csv(filepath: str, kwh_col: str = 'kWh', time_col: str = 'timestamp',
             seq_col: str = 'device_id', batch_size: int = None, limit_rows: int = -1):
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
            device_length = change_ids[i+1] - change_ids[i]
            seq_tags = numpy.concatenate([[str(data[seq_col][change_ids[i]]) + '_' + str(k + 1)
                                           for _ in range(0, batch_size)]
                                          for k in range(0, int(device_length / batch_size) + 1)],
                                         axis=None)
            data[seq_col][change_ids[i]:change_ids[i+1]] = seq_tags[0:device_length]
    data = data[[kwh_col, time_col, seq_col]]
    split_idx = [v for v in data[(data[seq_col].shift() != data[seq_col]) != 0].index if v > 0]

    # Create Sequences
    sequences = []
    for seq_data in numpy.split(data, split_idx):
        seq_data.reset_index(inplace=True)
        seq_tag = dataset_name + '_' + str(seq_data[seq_col][0])
        sequences.append(TimeSeriesSequence(seq_tag, numpy.asarray(seq_data[time_col]),
                                            numpy.asarray(seq_data[kwh_col]), None))

    return sequences


def load_dataset_config(cfg_file):
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
