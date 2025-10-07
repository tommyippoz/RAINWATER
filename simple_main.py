import copy
import os.path

import numpy

from rainwater import get_injectors
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
                    if new_s is not None and len(numpy.unique(new_s.labels)) >= 2:
                        injected_sequences.append(new_s)

            print('\nComputing Additional Features ...')
            for seq in injected_sequences:
                seq.add_features()

            # Prints a file for each dataset
            print('\nPrinting Data ...')
            print_full_sequences("my_file.csv", injected_sequences, create_new=True)

            # Performs model training
            model, stats, preds = model_trainer.train(injected_sequences,
                                                      dataset_name="my dataset",
                                                      models_folder=general_cfg['models_folder'],
                                                      save=False, debug=True)

            # prints csv file containing all stats
            dataset_name = cfg_file.replace(".cfg", "")
            with open(os.path.join(general_cfg['output_folder'],
                                   dataset_name + ("_binary" if general_cfg['force_binary'] else "_multi") +
                                   '_scores.csv'), 'w') as csvfile:

                # Print Header
                csvfile.write('policy,')
                stat = stats[list(stats.keys())[0]][0]
                for key in stat.keys():
                    if isinstance(stat[key], dict):
                        for dict_key in stat[key]:
                            if isinstance(stat[key][dict_key], dict):
                                for inner_key in stat[key][dict_key]:
                                    if not isinstance(stat[key][dict_key][inner_key], list):
                                        csvfile.write(key + "." + dict_key + "." + inner_key + ",")
                            elif not isinstance(stat[key][dict_key], list):
                                csvfile.write(key + "." + dict_key + ",")
                    elif not isinstance(stat[key], list):
                        csvfile.write(key + ",")
                csvfile.write('\n')

                # Print Data
                for policy in stats:
                    for stat in stats[policy]:
                        txt_row = policy + ","
                        for key in stat.keys():
                            if isinstance(stat[key], dict):
                                for dict_key in stat[key]:
                                    if isinstance(stat[key][dict_key], dict):
                                        for inner_key in stat[key][dict_key]:
                                            if not isinstance(stat[key][dict_key][inner_key], list):
                                                txt_row += str(stat[key][dict_key][inner_key]) + ","
                                    elif not isinstance(stat[key][dict_key], list):
                                        txt_row += str(stat[key][dict_key]) + ","
                            elif not isinstance(stat[key], list):
                                txt_row += str(stat[key]) + ","
                        csvfile.write(txt_row + '\n')
