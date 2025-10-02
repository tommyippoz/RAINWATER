import os.path

import numpy
import sklearn
from sklearn.preprocessing import LabelEncoder

from rainwater import get_injectors, sequences_to_series_dataset
from rainwater.LSTMClassifier import LSTMClassifier
from rainwater.ModelTrainer import ModelTrainer
from rainwater.utils import read_csv, load_dataset_config, print_full_sequences, read_general_conf

if __name__ == '__main__':

    # Reading general info
    general_cfg = read_general_conf('general_cfg.cfg')

    # Set the model trainer once
    model_trainer = ModelTrainer(policy=general_cfg['policy'],
                                 tt_split=general_cfg['tt_split'],
                                 force_binary=general_cfg['force_binary'])

    # Iterating over input files / case studies
    for cfg_file in os.listdir(general_cfg['input_folder']):

        if cfg_file.endswith('.cfg'):

            dataset_info = load_dataset_config(os.path.join(general_cfg['input_folder'], cfg_file))
            sequences = read_csv(dataset_info['csv_path'], dataset_info['kwh_col'],
                                 dataset_info['time_col'], dataset_info['seq_col'],
                                 dataset_info['batch_size'], dataset_info['limit_rows'])
            print(dataset_info['csv_path'])

            # Injecting anomalies
            injected_sequences = []
            print('\nPerforming Injection...')
            for injector in get_injectors(dataset_info['threats'], dataset_info['duration'],
                                          dataset_info['min_normal'], dataset_info['perc_inj']):
                for seq in sequences:
                    new_s = injector.inject(seq)
                    if new_s is not None and len(numpy.unique(new_s.labels)) >= 2 and len(injected_sequences) < 100:
                        injected_sequences.append(new_s)


            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, test_size=0.4)
            print("OK")
            model = LSTMClassifier()
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            print(sklearn.metrics.balanced_accuracy_score(preds, y_test))
            print("BOH")
