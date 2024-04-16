import copy
import json
import os
import random

import joblib
import numpy
import pandas
import sklearn.ensemble
from logitboost import LogitBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from rainwater import DecisionPolicy, sequences_to_dataset, current_ms, compute_stats
from rainwater.DecisionPolicy import policy_to_string, apply_policy


class ModelTrainer:

    def __init__(self, policy: list = [DecisionPolicy.NONE], tt_split: float = 0.5,
                 force_binary: bool = False, clf_list: list = None):
        self.policies = policy
        self.tt_split = tt_split
        self.force_binary = force_binary
        if clf_list is not None:
            self.classifiers = clf_list
        else:
            self.classifiers = [
                LinearDiscriminantAnalysis(),
                XGB(),
                DecisionTreeClassifier(),
                Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
                RandomForestClassifier(n_estimators=30),
                ExtraTreesClassifier(n_estimators=30),
                LogitBoost(n_estimators=30)
            ]

    def train(self, sequences: list, diagnosis_time: int = 1,  models_folder: str = None,
              dataset_name: str = None, save: bool = False, debug: bool = False):
        """
        Trains a model for a specific dataset of sequences
        :param debug: True if debug information has to be shown
        :param sequences: the sequences to be used for train/test
        :return: the trained model and the (performance) stats
        """
        model = None
        train_seq = random.sample(sequences, int(len(sequences)*self.tt_split))
        test_seq = list(set(sequences) - set(train_seq))
        train_x, train_y = sequences_to_dataset(train_seq, force_binary=self.force_binary)
        test_x, test_y = sequences_to_dataset(test_seq, force_binary=self.force_binary)
        if debug:
            print('%d train sequences (%d data points) and %d test sequences (%d data points)'
                  % (len(train_seq), len(train_y), len(test_seq), len(test_y)))

        # Loop over classifiers
        stats = {}
        preds = pandas.DataFrame()
        for clf in self.classifiers:
            ad = copy.deepcopy(clf)
            start_ms = current_ms()
            ad.fit(train_x.to_numpy(), train_y)
            train_ms = current_ms()
            ad_y = ad.predict(test_x.to_numpy())
            test_ms = current_ms()
            preds[ad.__class__.__name__] = ad_y
            if debug:
                print('Classifier %s, accuracy %.3f' % (ad.__class__.__name__,
                                                        sklearn.metrics.balanced_accuracy_score(ad_y, test_y)))
            for policy in self.policies:
                policy_str = policy_to_string(policy)
                policy_y = apply_policy(ad_y, policy)
                preds[ad.__class__.__name__ + "_" + policy_str] = policy_y

                # Computing Stats
                ad_stats = compute_stats(ad_y, policy_y, test_y, test_seq, diagnosis_time)
                ad_stats['diagnosis_time'] = diagnosis_time
                ad_stats['train_time'] = train_ms - start_ms
                ad_stats['test_time'] = test_ms - train_ms
                ad_stats['clf_name'] = ad.__class__.__name__
                ad_stats['train_seqs'] = len(train_seq)
                ad_stats['test_seqs'] = len(test_seq)
                ad_stats['train_datapoints'] = len(train_y)
                ad_stats['test_datapoints'] = len(test_y)
                joblib.dump(ad, "clf_dump.bin", compress=9)
                ad_stats['model_size_bytes'] = os.stat("clf_dump.bin").st_size
                os.remove("clf_dump.bin")
                if policy_str not in stats:
                    stats[policy_str] = []
                stats[policy_str].append(ad_stats)
                if debug:
                    print('\tPolicy %s, accuracy %.3f, avg_fpr %.3f, avg_tpr %.3f, avg_dd %.3f' %
                          (policy_str, ad_stats['ad']['accuracy'], ad_stats['ad']['s_fpr']['avg'],
                           ad_stats['ad']['s_tpr']['avg'], ad_stats['ad']['s_dd']['avg']))
                if save:
                    # Saving model to file
                    clf_type = "binary" if self.force_binary else "multi"
                    if not os.path.exists(os.path.join(models_folder, dataset_name)):
                        os.mkdir(os.path.join(models_folder, dataset_name))
                    base_folder = os.path.join(models_folder, dataset_name, clf_type)
                    if not os.path.exists(base_folder):
                        os.mkdir(base_folder)

                    filename_tag = dataset_name + "_" + ad_stats['clf_name'] + "_" + policy_str + "_" + clf_type
                    model_file = os.path.join(base_folder, filename_tag + ".joblib")
                    joblib.dump(ad, model_file, compress=9)

                    # Saving Stats to JSON file
                    stats_file = os.path.join(base_folder, filename_tag + "_stats.json")
                    json.dump(ad_stats, open(stats_file, 'w'), indent=4)

        preds = test_x.join(preds)
        preds['label'] = test_y

        return model, stats, preds


class XGB:
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self, n_estimators=100):
        self.clf = XGBClassifier(n_estimators=n_estimators)
        self.l_encoder = None
        self.classes = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        self.l_encoder = LabelEncoder()
        y = self.l_encoder.fit_transform(y)
        self.classes = self.l_encoder.classes_

        # Train clf
        self.clf.fit(X, y)

        # Return the classifier
        return self.clf

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.clf.predict_proba(X)
        return self.classes[numpy.argmax(probas, axis=1)]

    def classifier_name(self):
        return "XGBClassifier"
