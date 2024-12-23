import os.path

from rainwater import get_injectors
from rainwater.ModelTrainer import ModelTrainer, UnknownModelTrainer
from rainwater.utils import read_csv, load_dataset_config, print_full_sequences, read_general_conf

if __name__ == '__main__':

    # Reading general info
    general_cfg = read_general_conf('general_cfg.cfg')

    # Set the model trainer
    # In case you want to use custom classifiers, just put them in a list and pass them through clf_list
    # Make sure that these objects expose a fit, predict_proba and predict method
    model_trainer = UnknownModelTrainer(policy=general_cfg['policy'],
                                        tt_split=general_cfg['tt_split'],
                                        clf_list=None)

    # Iterating over input files looking for the RS2DG data loader
    for cfg_file in os.listdir(general_cfg['input_folder']):

        if not cfg_file.endswith('.cfg'):
            print('File %s is not a configuration file' % cfg_file)

        elif 'RS2DGGFDS' in cfg_file:
            print('The data loader \'%s\' does not belong to RS2DG case study' % cfg_file)

        else:
            # Read Sequences from Dataset
            print('\n---------------------------------------------\nReading dataset \'%s\'' % cfg_file)
            dataset_name = cfg_file.replace(".cfg", "")
            dataset_info = load_dataset_config(os.path.join(general_cfg['input_folder'], cfg_file))
            sequences = read_csv(dataset_info['csv_path'], dataset_info['kwh_col'],
                                 dataset_info['time_col'], dataset_info['seq_col'],
                                 dataset_info['batch_size'], dataset_info['limit_rows'])

            # Injecting anomalies A1 to A6
            injected_sequences = []
            print('\nPerforming Injection...')
            for injector in get_injectors(dataset_info['threats'], dataset_info['duration'],
                                          dataset_info['min_normal'], dataset_info['perc_inj']):
                for seq in sequences:
                    new_s = injector.inject(seq)
                    if new_s is not None:
                        injected_sequences.append(new_s)

            print('\nComputing Additional Features ...')
            for seq in injected_sequences:
                seq.add_features()

            # Prints a file for each dataset
            print('\nPrinting Data ...')
            dataset_file = os.path.join(general_cfg['output_folder'],
                                        general_cfg['data_tag'] + "_" + dataset_name + ".csv")
            print_full_sequences(dataset_file, injected_sequences, create_new=True)

            # Performs model training
            dict_out = model_trainer.train(injected_sequences, dataset_name=dataset_name,
                                           models_folder=general_cfg['models_folder'],
                                           save=True, debug=True)

            # Prints predictions
            preds_file = os.path.join(general_cfg['output_folder'], dataset_name + "_binary_predictions.csv")
            dict_out["normal"][2].to_csv(preds_file, index=False)

            # prints csv file containing all stats
            with open(os.path.join(general_cfg['output_folder'], dataset_name + '_unknown_scores.csv'), 'w') as csvfile:

                # Prints Header
                normal_stats = dict_out["normal"][1]
                csvfile.write('tag,policy,')
                stat = normal_stats[list(normal_stats.keys())[0]][0]
                for key in stat.keys():
                    if isinstance(stat[key], dict):
                        for dict_key in stat[key]:
                            if isinstance(stat[key][dict_key], dict):
                                for inner_key in stat[key][dict_key]:
                                    if isinstance(stat[key][dict_key][inner_key], dict):
                                        for inner_inner_key in stat[key][dict_key][inner_key]:
                                            if not isinstance(stat[key][dict_key][inner_key][inner_inner_key], list):
                                                csvfile.write(
                                                    key + "." + dict_key + "." + inner_key + "." + inner_inner_key + ",")
                                    elif not isinstance(stat[key][dict_key][inner_key], list):
                                        csvfile.write(key + "." + dict_key + "." + inner_key + ",")
                            elif not isinstance(stat[key][dict_key], list):
                                csvfile.write(key + "." + dict_key + ",")
                    elif not isinstance(stat[key], list):
                        csvfile.write(key + ",")
                csvfile.write('\n')

                # Prints Data
                for tag in dict_out:
                    stats = dict_out[tag][1]
                    for policy in stats:
                        for stat in stats[policy]:
                            txt_row = tag + "," + policy + ","
                            for key in stat.keys():
                                if isinstance(stat[key], dict):
                                    for dict_key in stat[key]:
                                        if isinstance(stat[key][dict_key], dict):
                                            for inner_key in stat[key][dict_key]:
                                                if isinstance(stat[key][dict_key][inner_key], dict):
                                                    for inner_inner_key in stat[key][dict_key][inner_key]:
                                                        if not isinstance(stat[key][dict_key][inner_key][inner_inner_key],
                                                                          list):
                                                            txt_row += str(
                                                                stat[key][dict_key][inner_key][inner_inner_key]) + ","
                                                elif not isinstance(stat[key][dict_key][inner_key], list):
                                                    txt_row += str(stat[key][dict_key][inner_key]) + ","
                                        elif not isinstance(stat[key][dict_key], list):
                                            txt_row += str(stat[key][dict_key]) + ","
                                elif not isinstance(stat[key], list):
                                    txt_row += str(stat[key]) + ","
                            csvfile.write(txt_row + '\n')
