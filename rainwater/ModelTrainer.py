import copy
import json
import os
import random

import joblib
import numpy
import pandas
import sklearn.ensemble
from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from logitboost import LogitBoost
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

from rainwater import DecisionPolicy, sequences_to_dataset, current_ms, compute_stats, get_classifier_name, \
    sequences_to_unknown_dataset
from rainwater.DecisionPolicy import policy_to_string, apply_policy
from rainwater.LSTMClassifier import LSTMClassifier


class ModelTrainer:
    """
    Class that exposes the methods for training a classifier for a problem.
    Compares a set of classifiers over a set of sequances, performing data augmentation in the process.
    The dataset has to have a feature that shows power consumption.
    """

    def __init__(self, policy: list = [DecisionPolicy.NONE], tt_split: float = 0.5,
                 force_binary: bool = False, clf_list: list = None):
        """
        Constructor
        :param policy: the DecisionPolicy(ies) to be used in a list
        :param tt_split: the percentage of sequences to be used for training (others for testing)
        :param force_binary: True if binary classification, multi-class otherwise
        :param clf_list: the list of candidate classifiers to be used.
        """
        self.policies = policy
        self.tt_split = tt_split
        self.force_binary = force_binary
        if clf_list is not None:
            self.classifiers = clf_list
        else:
            self.classifiers = [
                LSTMClassifier(force_binary=force_binary),
                LinearDiscriminantAnalysis(),
                QuadraticDiscriminantAnalysis(),
                XGB(),
                DecisionTreeClassifier(),
                Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
                RandomForestClassifier(),
                ExtraTreesClassifier(),
                ConfidenceBagging(clf=ExtraTreeClassifier()),
                ConfidenceBoosting(clf=ExtraTreeClassifier()),

            ]
            if force_binary:
                contamination = 0.2
                self.classifiers.extend([UnsupervisedClassifier(COPOD(contamination=contamination)),
                                         UnsupervisedClassifier(HBOS(contamination=contamination)),
                                         UnsupervisedClassifier(CBLOF(contamination=contamination)),
                                         UnsupervisedClassifier(IForest(contamination=contamination)),
                                         ConfidenceBagging(clf=HBOS(contamination=contamination)),
                                         ConfidenceBoosting(clf=HBOS(contamination=contamination))])

    def train(self, sequences: list, diagnosis_time: int = 1, models_folder: str = None,
              dataset_name: str = None, save: bool = False, debug: bool = False):
        """
        Trains a model for a specific dataset of sequences
        :param clean_sequences: clean sequences, no additional features
        :param save: True if the model and stats have to be saved to joblib and JSON files
        :param dataset_name: the name/tag of the dataset
        :param models_folder: the folder used to load/store models
        :param diagnosis_time: placeholder - not implemented yet
        :param debug: True if debug information has to be shown
        :param sequences: the sequences to be used for train/test
        :return: the trained model and the (performance) stats
        """
        train_seq = random.sample(sequences, int(len(sequences) * self.tt_split))
        test_seq = list(set(sequences) - set(train_seq))
        train_x, train_y = sequences_to_dataset(train_seq, force_binary=self.force_binary)
        test_x, test_y = sequences_to_dataset(test_seq, force_binary=self.force_binary)
        if debug:
            print('%d train sequences (%d data points) and %d test sequences (%d data points)'
                  % (len(train_seq), len(train_y), len(test_seq), len(test_y)))

        return self.train_model(train_seq, test_seq, train_x, train_y, test_x, test_y,
                                diagnosis_time,
                                models_folder, dataset_name, save, debug)

    def train_model(self, train_seq, test_seq, train_x, train_y, test_x, test_y,
                    diagnosis_time: int = 1,
                    models_folder: str = None, dataset_name: str = None, save: bool = False,
                    debug: bool = False):
        """
        Trains a model for a specific dataset of sequences
        :param save: True if the model and stats have to be saved to joblib and JSON files
        :param dataset_name: the name/tag of the dataset
        :param models_folder: the folder used to load/store models
        :param diagnosis_time: placeholder - not implemented yet
        :param debug: True if debug information has to be shown
        :return: the trained model and the (performance) stats
        """
        # Loop over classifiers
        model = None
        stats = {}
        preds = pandas.DataFrame()
        for clf in self.classifiers:
            ad = copy.deepcopy(clf)
            ad_name = get_classifier_name(ad)
            start_ms = current_ms()
            if isinstance(clf, LSTMClassifier):
                ad.fit(train_seq)
                train_ms = current_ms()
                ad_y = ad.predict(test_seq)
            else:
                ad.fit(train_x.to_numpy(), train_y)
                train_ms = current_ms()
                ad_y = ad.predict(test_x.to_numpy())
            test_ms = current_ms()
            preds[ad_name] = ad_y
            if debug:
                print('Classifier %s, accuracy %.3f' % (ad_name,
                                                        sklearn.metrics.balanced_accuracy_score(test_y, ad_y)))
            for policy in self.policies:
                policy_str = policy_to_string(policy)
                policy_y = apply_policy(ad_y, policy)
                preds[ad_name + "_" + policy_str] = policy_y

                # Computing Stats
                ad_stats = compute_stats(ad_y, policy_y, test_y, test_seq, diagnosis_time)
                ad_stats['diagnosis_time'] = diagnosis_time
                ad_stats['train_time'] = train_ms - start_ms
                ad_stats['test_time'] = test_ms - train_ms
                ad_stats['clf_name'] = ad_name
                ad_stats['train_seqs'] = len(train_seq)
                ad_stats['test_seqs'] = len(test_seq)
                ad_stats['train_datapoints'] = len(train_y)
                ad_stats['test_datapoints'] = len(test_y)
                # joblib.dump(ad, "clf_dump.bin", compress=9)
                ad_stats['model_size_bytes'] = -1  # os.stat("clf_dump.bin").st_size
                # os.remove("clf_dump.bin")
                if policy_str not in stats:
                    stats[policy_str] = []
                stats[policy_str].append(ad_stats)
                if debug:
                    print('\tPolicy %s, accuracy %.3f, avg_fpr %.3f, avg_tpr %.3f, avg_dd %.3f' %
                          (policy_str, ad_stats['ad']['accuracy'], ad_stats['ad']['overall_fpr']['avg'],
                           ad_stats['ad']['overall_tpr']['avg'], ad_stats['ad']['overall_dd']['avg']))
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


class UnknownModelTrainer(ModelTrainer):
    """
    Class that exposes the methods for training a classifier for a problem.
    Compares a set of classifiers over a set of sequances, performing data augmentation in the process.
    The dataset has to have a feature that shows power consumption.
    """

    def __init__(self, policy: list = [DecisionPolicy.NONE], tt_split: float = 0.5, clf_list: list = None):
        """
        Constructor
        :param policy: the DecisionPolicy(ies) to be used in a list
        :param tt_split: the percentage of sequences to be used for training (others for testing)
        :param clf_list: the list of candidate classifiers to be used.
        """
        super().__init__(policy, tt_split, True, clf_list)

    def train(self, sequences: list, diagnosis_time: int = 1, models_folder: str = None,
              dataset_name: str = None, save: bool = False, debug: bool = False):
        """
        Trains a model for a specific dataset of sequences
        :param save: True if the model and stats have to be saved to joblib and JSON files
        :param dataset_name: the name/tag of the dataset
        :param models_folder: the folder used to load/store models
        :param diagnosis_time: placeholder - not implemented yet
        :param debug: True if debug information has to be shown
        :param sequences: the sequences to be used for train/test
        :return: the trained model and the (performance) stats for all combinations of unknowns as dict
        """
        train_seq = random.sample(sequences, int(len(sequences) * self.tt_split))
        test_seq = list(set(sequences) - set(train_seq))
        train_data = sequences_to_unknown_dataset(train_seq, train_flag=True)
        test_data = sequences_to_unknown_dataset(test_seq, train_flag=False)
        if debug:
            print('%d train sequences and %d test sequences ' % (len(train_seq), len(test_seq)))
        results = {}
        for tag in train_data:
            if debug:
                print("\n---------------------- Processing unknown tag: %s ------------------------------\n" % tag)
            model, stats, preds = self.train_model(train_data[tag][2], test_data[tag][2], train_data[tag][0],
                                                   train_data[tag][1],
                                                   test_data[tag][0], test_data[tag][1], diagnosis_time,
                                                   models_folder, dataset_name, save, debug)
            results[tag] = [model, stats, preds]
        return results


class XGB:
    """
    Wrapper for the xgboost.XGBClassifier algorithm to overcome the problem with string labels
    """

    def __init__(self, n_estimators=100):
        """
        Constructor
        :param n_estimators: number of estimators
        """
        self.clf = XGBClassifier(n_estimators=n_estimators)
        self.l_encoder = None
        self.classes = None

    def fit(self, X, y=None):
        """
        Trains the classifier
        :param X: train features
        :param y: train labels
        :return: the trained model
        """

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
        :param X: test features
        :return: array of predicted class
        """
        probas = self.clf.predict_proba(X)
        return self.classes[numpy.argmax(probas, axis=1)]

    def classifier_name(self):
        """
        Returns the name of the classifier
        :return: a string
        """
        return "XGBClassifier"


class UnsupervisedClassifier:
    """
    Wrapper for the xgboost.XGBClassifier algorithm to overcome the problem with string labels
    """

    def __init__(self, uns_clf):
        """
        Constructor
        :param n_estimators: number of estimators
        """
        self.clf = copy.deepcopy(uns_clf)

    def fit(self, X, y=None):
        """
        Trains the classifier
        :param X: train features
        :param y: train labels
        :return: the trained model
        """
        # Train clf
        self.clf.fit(X)

        # Return the classifier
        return self.clf

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :param X: test features
        :return: array of predicted class
        """
        num_pred = numpy.asarray(self.clf.predict(X))
        return numpy.where(num_pred == 0, "normal", "anomaly")

    def classifier_name(self):
        """
        Returns the name of the classifier
        :return: a string
        """
        return self.clf.__class__.__name__
