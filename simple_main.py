import os.path

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
            sequences = read_csv(dataset_info)

            # Injecting anomalies
            injected_sequences = []
            print('\nPerforming Injection...')
            for injector in get_injectors(dataset_info):
                for seq in sequences:
                    new_s = injector.inject(seq)
                    if new_s is not None:
                        injected_sequences.append(new_s)

            print('\nComputing Additional Features ...')
            for seq in injected_sequences:
                seq.add_features()

            # Prints a file for each dataset
            print('\nPrinting Data ...')
            print_full_sequences("my_file.csv", injected_sequences, create_new=True)

            # Performs model training
            model, stats, preds = model_trainer.train(injected_sequences, dataset_name="my dataset",
                                                      models_folder=general_cfg['models_folder'],
                                                      save=True, debug=True)

