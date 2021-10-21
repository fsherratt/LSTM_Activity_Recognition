import collections
import copy
import math
import os
import pathlib
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf


class InsufficientData(Exception):
    """
    There is insufficient data to complete the operation - reduce data requirements
    """


class Data_Files:
    def __init__(
        self, data_folder: str, verbose=True, shuffle=False, preload=True,
    ):
        self.verbose = verbose

        self.extension = ".csv"
        self.file_dict = {}

        self.file_list = self.get_file_list(data_folder, self.extension)

        if shuffle:
            random.seed(27)  # was 1
            random.shuffle(self.file_list)

        if preload:
            self.preload_data()

    def find_cache_data(self, data_file):
        if data_file not in self.file_dict:
            if self.verbose:
                print(f"Loading {data_file.name}")
            self.file_dict[data_file] = self.load_file(data_file)

        return self.file_dict[data_file]

    def preload_data(self):
        for file_name in self.file_list:
            if self.load_augment or not self.is_augmented_data(str(file_name)):
                self.file_dict[file_name] = self.load_file(file_name)

    def exclude_filter(self, filter_dir: str) -> list:
        if filter_dir is None:
            return self.file_list

        return [
            file_name
            for file_name in self.file_list
            if self.check_if_not_in_dir(str(file_name), filter_dir)
        ]

    def include_filter(self, filter_dir: str) -> list:
        if filter_dir is None:
            return self.file_list

        return [
            file_name
            for file_name in self.file_list
            if not self.check_if_not_in_dir(str(file_name), filter_dir)
        ]

    @staticmethod
    def check_if_not_in_dir(file_name: str, dirs: list) -> bool:
        """
        Check if the file path contains in the directory list

        @param file_name: str
            File name to test

        @param dirs: list
            List of directory names to test

        @return bool
            True if not in list of directories, False otherwise
        """
        return all(exclude not in file_name for exclude in dirs)

    def _gen_activity_file_lists(self, data_files: list) -> dict:
        """
        Seperate out the different file types into seperate lists

        @param data_files: list
            A list of data file Path objects

        @return dict
            Dictionary of lists - one list for each activity type
        """
        activity_list = collections.defaultdict(list)

        for file in data_files:
            activity_list[self.get_activity_type(file.name)].append(file)

        return dict(activity_list)

    @staticmethod
    def _offset_file_dict(data_files: dict, offset: int) -> dict:
        """
        Offset all of the lists in a dictionary

        @param data_files: dict
            Dictionary of lists - one list for each activity type

        @param offset: int
            Amount of entries to offset the list by

        @return dict
            Dictionary of lists - one list for each activity type
        """
        for key, value in data_files.items():
            if offset > len(value):
                raise InsufficientData(
                    f"The number of data files for '{key}' is less than offset {offset}"
                )

            value = value[offset:] + value[:offset]

        return data_files

    @staticmethod
    def _clip_per_activity_data_windows(data_dict: dict, max_items: int) -> dict:
        """
        Ensure each activity has no more that the maximum items, discard items randomly

        @param data_dict: dict
            Dictionary of lists for each activity

        @param max_items: int
            Maximum number of items to cap at

        @return dict
            Capped dictionary of lists for each activity
        """
        for key, value in data_dict.items():
            perms = np.random.permutation(len(value[1]))
            perms = list(perms[:max_items])

            data = [value[0][i] for i in perms]
            labels = [value[1][i] for i in perms]

            data_dict[key] = (data, labels)

        return data_dict

    def _parse_episode_list(
        self, input_data_files: dict, target_samples: int, parse_settings: dict
    ) -> tuple:
        """
        Parse the episode list until the correct amount of samples has been reached. A IndexError is raised if not enough data available

        @param input_data_files : dict
             Dictionary of lists - one list for each activity type

        @param target_samples: int
            Target number of data samples

        @param parse_settings: dict
            Dictionary of parsing settings
        """
        activity_data_files = copy.copy(input_data_files)

        activity_samples = collections.defaultdict(lambda: ([], []))
        for key, value in activity_data_files.items():
            while len(activity_samples[key][1]) < target_samples:
                try:
                    file = value.pop(0)
                except IndexError:
                    raise InsufficientData(f"Not enough data for activity {key}")

                x, y = self.parse_file(self.find_cache_data(file), **parse_settings)

                activity_samples[key][0].extend(x)
                activity_samples[key][1].extend(y)

        return dict(activity_samples), activity_data_files

    def load_data_episodes(
        self,
        parse_kwargs: dict,
        file_offset: int,
        test_samples_min: int,
        train_samples_min: int,
        valid_samples_min: int,
        filter_mode=True,  # True = exclude items in list, False = only load list
        filter_list=None,  # List
    ) -> dict:
        """
        Parse data files for episode based data

        @param parse_kwargs: dict
            Keyword arguments for the data parser

        @param file_offset: int
            The starting point in the data file list - allows for consistent shuffling of data

        @param test_samples_min: int
            The minimum number of test samples to return

        @param train_samples_min: int
            The minimum number of training samples to return

        @param valid_samples_min: int
            The minimum number of validation samples to return

        @filter mode: bool
            True if items in the filter list directories should be excluded, False if only filter list directories should be included

        @return dict of train, test and validation data dictionaries
        """

        if filter_mode is None:
            data_files = self.file_list
        elif filter_mode:
            data_files = self.exclude_filter(filter_list)
        else:
            data_files = self.include_filter(filter_list)

        activity_file_dict = self._gen_activity_file_lists(data_files)
        activity_file_dict = self._offset_file_dict(activity_file_dict, file_offset)

        # Parse episodes until the correct amount of data is obtained
        # Test data
        test_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, test_samples_min, parse_kwargs
        )

        # Train data
        train_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, train_samples_min, parse_kwargs
        )

        # Validataion data
        valid_data, activity_file_dict = self._parse_episode_list(
            activity_file_dict, valid_samples_min, parse_kwargs
        )

        return {"train": train_data, "valid": valid_data, "test": test_data}

    def load_data_activities(
        self,
        parse_kwargs: dict,
        file_offset: int,
        test_samples: int,
        train_samples: int,
        valid_samples: int,
        filter_mode=True,
        filter_list=None,
    ):
        """
        Extends load data episodes by adding left/right ankle, clipping max data, combining activities

        @param parse_kwargs: dict
            Keyword arguments for the data parser

        @param file_offset: int
            The starting point in the data file list - allows for consistent shuffling of data

        @param test_sample: int
            The number of test samples to return

        @param train_sample: int
            The number of training samples to return

        @param valid_sample: int
            The number of validation samples to return

        @filter mode: bool
            True if items in the filter list directories should be excluded, False if only filter list directories should be included

        @return dict of train, test and validation data dictionaries
        """
        data_episodes_kwargs = {
            "parse_kwargs": parse_kwargs,
            "file_offset": file_offset,
            "test_samples_min": int(test_samples / 2),
            "train_samples_min": int(train_samples / 2),
            "valid_samples_min": int(valid_samples / 2),
            "filter_mode": filter_mode,
            "filter_list": filter_list,
        }

        r_parse_kwargs = copy.copy(parse_kwargs)
        r_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("r")
        ]

        l_parse_kwargs = copy.copy(parse_kwargs)
        l_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("l")
        ]

        # Load data for left and right ankles
        data_episodes_kwargs["parse_kwargs"] = r_parse_kwargs
        r_data = self.load_data_episodes(**data_episodes_kwargs)

        data_episodes_kwargs["parse_kwargs"] = l_parse_kwargs
        l_data = self.load_data_episodes(**data_episodes_kwargs)

        # Clip data
        for key, value in r_data.items():
            r_data[key] = self._clip_per_activity_data_windows(
                value, data_episodes_kwargs[key + "_samples_min"]
            )

        for key, value in l_data.items():
            l_data[key] = self._clip_per_activity_data_windows(
                value, data_episodes_kwargs[key + "_samples_min"]
            )

        # Combine into one data set
        data = activity_samples = collections.defaultdict(lambda: ([], []))

        for key, value in r_data.items():
            for activity_value in value.values():
                data[key][0].extend(activity_value[0])
                data[key][1].extend(activity_value[1])

        for key, value in l_data.items():
            for activity_value in value.values():
                data[key][0].extend(activity_value[0])
                data[key][1].extend(activity_value[1])

        # Convert to np.array
        for key, value in data.items():
            data[key] = (np.asarray(value[0]), value[1])

        return dict(data)

    def load_data(
        self,
        parse_kwargs: dict,
        validation_split: float,
        filter_mode=True,
        filter_list=None,
        append=False,
    ):
        """
        Load all data, return split

        @param parse_kwargs: dict
            keyword arguments to pass to the parser

        @param validation_split: float
            Percentage of data to use for validation

        @param filter_mode: bool
            True = exclude items in list, False = only load list

        @param filter_list: list[str]
            List of folders to exclude for the loaded data
        
        @param append: bool
            Should the loaded date be appended or extended

        @return: dict
            Dictionary of loaded data - returned in the following format
            {"train": np.array,
             "valid: np.array,
             "test: None}

        TODO: Add in both left and right ankle
        """
        x = []
        y = []

        if filter_mode:
            data_files = self.exclude_filter(filter_list)
        else:
            data_files = self.include_filter(filter_list)

        r_parse_kwargs = copy.copy(parse_kwargs)
        r_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("r")
        ]

        l_parse_kwargs = copy.copy(parse_kwargs)
        l_parse_kwargs["data_headings"] = [
            i for i in parse_kwargs["data_headings"] if i.startswith("l")
        ]

        for file in data_files:

            if self.verbose:
                print("Loaded file {}".format(str(file)))

            # Load right ankle
            new_x, new_y = self.parse_file(self.find_cache_data(file), **r_parse_kwargs)

            if append:
                x.append(new_x)
                y.append(new_y)
            else:
                x.extend(new_x)
                y.extend(new_y)

            # Load left ankle
            new_x, new_y = self.parse_file(self.find_cache_data(file), **l_parse_kwargs)

            if append:
                x.append(new_x)
                y.append(new_y)
            else:
                x.extend(new_x)
                y.extend(new_y)

        x = np.asarray(x)
        y = np.asarray(y)

        train_data, valid_data = self.split_test_validation(x, y, validation_split)

        return {"train": train_data, "valid": valid_data, "test": None}

    @staticmethod
    def get_activity_type(file_name: str) -> tuple:
        """
        Search file name to try and identify activity and split index

        @param file_name
            In format ABC_freq_ACTIVITY_NAME_Log_Date_time.csv

        @return: str 
            activity, None is match not found
        """
        re_output = re.search(r"[\w]*_\d*_([\w]*)_Log_(\d*)_(\d*)_(\d*).csv", file_name)

        if not re_output:
            return None

        file_activity = re_output.group(1)
        file_date = re_output.group(2)
        file_time = re_output.group(3)
        file_seq = re_output.group(4)

        return file_activity

    @staticmethod
    def get_file_list(data_directory: str, extension: str) -> list:
        data_directory = pathlib.Path(data_directory)
        files = os.listdir(data_directory)
        data_files = []

        for file in files:
            file = data_directory / file

            if os.path.isdir(file):
                data_files.extend(Data_Files.get_file_list(file, extension))

            elif file.exists() and file.suffix == extension:
                data_files.append(file)

        return data_files

    @staticmethod
    def load_file(file) -> pd.core.frame.DataFrame:
        return pd.read_csv(file)

    @staticmethod
    def parse_file(
        data,
        label_heading,
        data_headings,
        num_timesteps,
        num_labels,
        skip,
        normalize,
        label_mapping=None,
    ):
        labels_data = data[label_heading]
        labels_data = np.asarray(labels_data, dtype=np.int32)
        input_data = data[data_headings]
        input_data = np.asarray(input_data, dtype=np.float32)

        # Output filter and mapping
        if label_mapping is not None:
            included_activites = np.asarray(list(label_mapping.keys()))

            # Filter
            filter_list = np.isin(labels_data, included_activites)
            labels_data = labels_data[filter_list]
            input_data = input_data[filter_list]

        # Apply one_hot to labels
        labels_data = np.asarray(labels_data, dtype=np.int8)
        input_data = np.asarray(input_data, dtype=np.float32)

        data_frames = []
        label_frames = []

        i = 0
        max_start_index = labels_data.shape[0] - num_timesteps

        while i < max_start_index:
            sample_end = i + num_timesteps

            data_frames.append(input_data[i:sample_end, :])
            label_frames.append(labels_data[sample_end])

            i += skip + 1

        # Normalise for each data window
        if normalize:
            data_frames = tf.keras.utils.normalize(data_frames, axis=1, order=2)

        return data_frames, label_frames

    @staticmethod
    def split_test_validation(
        data, labels, split: float,
    ):
        """
        Split data into test and validation

        Ensure there is an equal amount of each label type of both data sets

        @param data: np.array
            Numpy array of data windows
        
        @param labels: np.array
            Numpy array of data labels

        @param split: float
            Percentage validation data

        @return: tuple
            (Training data, Validation data)
        """
        train_rows = []
        valid_rows = []

        # Do something to limit max difference between labels
        unique_labels, label_count = np.unique(labels, return_counts=True)

        # TODO: add in the ability to limit the quantity of each label category

        min_label_count = np.min(label_count)

        if min_label_count == 0:
            raise RuntimeError("Min count is zero")

        for i in range(unique_labels.shape[0]):
            valid_entries = math.floor(split * min_label_count)
            # valid_entries = int(split * label_count[i])
            label_rows = np.where(labels == unique_labels[i])[0]

            perms = np.random.permutation(label_count[i])

            train_rows.extend(label_rows[perms[:valid_entries]])
            valid_rows.extend(label_rows[perms[valid_entries:min_label_count]])

        print(label_count)
        train = (
            data.take(train_rows, axis=0),
            labels.take(train_rows, axis=0),
        )
        valid = (
            data.take(valid_rows, axis=0),
            labels.take(valid_rows, axis=0),
        )

        return (
            train,
            valid,
        )

