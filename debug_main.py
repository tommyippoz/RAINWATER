import os.path

import numpy
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from rainwater import get_injectors
from rainwater.utils import read_csv, load_dataset_config, print_sequences

INPUT_FOLDER = 'input'
SEQ_FILE = 'preprocessed/preproc_all.csv'

if __name__ == '__main__':

    create_file = True
    for cfg_file in os.listdir(INPUT_FOLDER):

        if not cfg_file.endswith('.cfg'):
            print('File %s is not a configuration file' % cfg_file)

        else:
            # Read Sequences from Dataset
            print('\n---------------------------------------------\nReading dataset \'%s\'' % cfg_file)
            dataset_info = load_dataset_config(os.path.join(INPUT_FOLDER, cfg_file))
            sequences = read_csv(dataset_info['csv_path'], dataset_info['kwh_col'],
                                 dataset_info['time_col'], dataset_info['seq_col'],
                                 dataset_info['batch_size'], dataset_info['limit_rows'])

            # Injecting anomalies
            injected_sequences = []
            for injector in get_injectors(dataset_info['threats'], dataset_info['duration'],
                                          dataset_info['min_normal'], dataset_info['perc_inj']):
                for seq in sequences:
                    injected_sequences.append(injector.inject(seq))

            #for seq in injected_sequences:



            # Prints a file for each dataset
            dataset_file = SEQ_FILE.replace(".csv", "_" + cfg_file).replace(".cfg", ".csv")
            print_sequences(dataset_file, injected_sequences, create_new=True)

            # Prints a file containing everything
            print_sequences(SEQ_FILE, injected_sequences, create_new=create_file)
            create_file = False






def other():
    # Read the train dataset
    data = pd.read_csv(TRAIN_FILE)
    y_train = data['label']
    x_train = data.drop(columns=['label', 'device_Id', 'timestamp_created'])
    x_train = x_train.to_numpy()

    # Read the test dataset
    data = pd.read_csv(TEST_FILE)
    test_split_indexes = [v for v in data[(data["device_Id"].shift() != data["device_Id"]) != 0].index if v > 0]
    device_ids = data['device_Id'][data[(data["device_Id"].shift() != data["device_Id"]) != 0].index].to_numpy()
    y_test = data['label']
    x_test = data.drop(columns=['label', 'device_Id', 'timestamp_created'])
    x_test = x_test.to_numpy()

    # Prepare test set for DD and Fpr
    x_test_batch = numpy.split(x_test, test_split_indexes)
    y_test_batch = numpy.split(y_test, test_split_indexes)

    # Exercise different classifiers (only decision tree to generate the model)
    print('Using %d rows for training, %d for testing' % (len(y_train), len(y_test)))

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)

    # Make predictions on the testing set and compute accuracy
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Array of DDs for each device series in the test set
    dd = []
    # Array of FPRs for each device series in the test set
    fpr = []
    # Lists to accumulate counts for confusion matrix
    true_labels = []
    pred_labels = []
    # Array of TPRs for each device series in the test set
    tprs= []
    # Results
    device_dict = {}

    # Compute DD and FPR
    index = 0
    for x_device in x_test_batch:
        y_pred_device = clf.predict(x_device)
        y_test_device = y_test_batch[index].to_numpy()


        # find first occurrence of anomaly in the series for that device
        anomaly_index = next((i for i in range(len(y_test_device)) if y_test_device[i] > 0), -1)

        dict_res = {}
        # Update fpr and DD
        if anomaly_index > 0:
            # It means that an anomaly exists in the series for the device
            fpr.append(sum(y_pred_device[:anomaly_index]*(1-y_test_device[:anomaly_index]))/(anomaly_index+1))
            tp = np.where((y_pred_device + y_test_device == 2))[0]
            dd.append((tp[0] - anomaly_index) if len(tp) > 0 else 1000)
            tpr = 1 if len(tp) > 0 else 0
            tprs.append(tpr)
            dict_res = {'tpr': tpr,
                        'fpr': sum(y_pred_device[:anomaly_index]*(1-y_test_device[:anomaly_index]))/(anomaly_index+1),
                        'dd': (tp[0] - anomaly_index) if len(tp) > 0 else 1000}
        else:
            # It means that no anomaly is found in the series for the device
            #fpr.append(-1)
            #dd.append(-1)
            dict_res = {'tpr': -1,
                        'fpr': -1,
                        'dd': -1}
            print('no anomaly was found')
        device_dict[device_ids[index]] = dict_res

        # Accumulate true and predicted labels for confusion matrix
        true_labels.extend(y_test_device)
        pred_labels.extend(y_pred_device)

        # Compute confusion matrix
        a = confusion_matrix(true_labels, pred_labels).ravel()
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()

        index += 1


    # Print Metrics and Confusion Matrix
    print("\n-------CLASSIFIER DecisionTree -------------")
    print("Confusion Matrix: [tn:%d, tp:%d, fn:%d, fp:%d], ACC:%.3f" % (tn, tp, fn, fp, accuracy))
    print("DD [Med, Avg, Max]: [%d, %.3f, %d] - all: (%s)" % (numpy.median(dd), numpy.average(dd), numpy.max(dd),
                                                       "; ".join([str(d) for d in dd])))
    print("FPR [Med, Avg]: [%.3f, %.3f] - all: (%s), TPR [Med, Avg]: [%d, %.3f]" %
          (numpy.median(fpr), numpy.average(fpr), "; ".join(["{:.3f}".format(d) for d in fpr]), numpy.median(tprs), numpy.average(tprs)))


    final_df = data[['lumped_kWh', 'device_Id', 'timestamp_created', 'label']]
    final_df['prediction_dt'] = pred_labels
    final_df.to_csv('debug_output_dt.csv', index=False)

    with open('metrics_all.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        f.write('device_id, tpr, fpr, dd\n')
        for res_d in device_dict.keys():
            f.write(res_d + ',' + str(device_dict[res_d]['tpr']) + ',' +
                    str(device_dict[res_d]['fpr']) + ',' + str(device_dict[res_d]['dd']) + "\n")

