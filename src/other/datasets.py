"""
The dataset preprocessing is based on DeepAR
https://arxiv.org/pdf/1704.04110.pdf
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sktime.datasets import load_from_arff_to_dataframe
from torch import Tensor
import os, os.path
import urllib.response
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchaudio.datasets import SPEECHCOMMANDS
import os
import urllib.request
import tarfile
import shutil
import librosa
import torch.utils.data as data
from scipy import integrate
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler



class SinusoidalDataset(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)
        
    
        # Add more noise to the signal
        #signal += 0.1 * np.random.randn(self.seq_length)

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    
class SinusoidalDatasetLong(Dataset):
    def __init__(self, num_samples, seq_length=100, num_features=1, freq_min=10, freq_max=500, num_classes=20):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sinusoidal signals with noise and trend
        frequencies = [i for i in range(self.freq_min, self.freq_max+1, (self.freq_max-self.freq_min)//self.num_classes)]
        freq = frequencies[idx % self.num_classes]

        t = np.linspace(0, 1, self.seq_length)
        signal = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)

        # Add a non-linearly increasing or decreasing trend
        trend = np.linspace(-0.5, 0.5, self.seq_length)
        if np.random.rand() < 0.5:
            trend = np.square(trend)
        else:
            trend = -np.square(trend) 
        signal += trend

        # Add more complex patterns to the signal
        signal += 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.sin(8 * np.pi * freq * t)

        # Add more noise to the signal
        signal += 0.1 * np.random.randn(self.seq_length)

        signal2 = 0.5 * np.sin(2 * np.pi * (freq / 5) * t + np.random.uniform(0, 2*np.pi)) + 0.1 * np.random.randn(self.seq_length)
        signal2 += 0.2 * np.sin(4 * np.pi * (freq / 5) * t) + 0.1 * np.sin(8 * np.pi * (freq / 5) * t)
        signal2 += 0.1 * np.random.randn(self.seq_length)

        signal[250:] = signal2[250:]

        label = frequencies.index(freq)
        sample = {'input': torch.tensor(signal, dtype=torch.float).view(-1, self.num_features), 'label': label}

        return sample
    

class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """
    pass


#Taken from https://github.com/ChangWeiTan/TSRegression/blob/master/utils/data_loader.py and https://github.com/tk-rusch/LEM/blob/main/src/heart_rate/data.py

class HeartRate():
    def __init__(self):
        pass

    def load_from_tsfile_to_dataframe(self, full_file_path_and_name, return_separate_X_and_y=True,
                                    replace_missing_vals_with='NaN'):
        """Loads data from a .ts file into a Pandas DataFrame.
        Parameters
        ----------
        full_file_path_and_name: str
            The full pathname of the .ts file to read.
        return_separate_X_and_y: bool
            true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
            This is only relevant for data that
        replace_missing_vals_with: str
        The value that missing values in the text file should be replaced with prior to parsing.
        Returns
        -------
        DataFrame, ndarray
            If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
        DataFrame
            If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
        """

        # Initialize flags and variables used when parsing the file
        metadata_started = False
        data_started = False

        has_problem_name_tag = False
        has_timestamps_tag = False
        has_univariate_tag = False
        has_class_labels_tag = False
        has_target_labels_tag = False
        has_data_tag = False

        previous_timestamp_was_float = None
        previous_timestamp_was_int = None
        previous_timestamp_was_timestamp = None
        num_dimensions = None
        is_first_case = True
        instance_list = []
        class_val_list = []
        line_num = 0

        # Parse the file
        # print(full_file_path_and_name)
        with open(full_file_path_and_name, 'r', encoding='ISO-8859-1') as file:
            for line in tqdm(file):
                # print(".", end='')
                # Strip white space from start/end of line and change to lowercase for use below
                line = line.strip().lower()
                # Empty lines are valid at any point in a file
                if line:
                    # Check if this line contains metadata
                    # Please note that even though metadata is stored in this function it is not currently published externally
                    if line.startswith("@problemname"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")
                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("problemname tag requires an associated value")

                        problem_name = line[len("@problemname") + 1:]
                        has_problem_name_tag = True
                        metadata_started = True
                    elif line.startswith("@timestamps"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len != 2:
                            raise TsFileParseException("timestamps tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            timestamps = True
                        elif tokens[1] == "false":
                            timestamps = False
                        else:
                            raise TsFileParseException("invalid timestamps value")
                        has_timestamps_tag = True
                        metadata_started = True
                    elif line.startswith("@univariate"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)
                        if token_len != 2:
                            raise TsFileParseException("univariate tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            univariate = True
                        elif tokens[1] == "false":
                            univariate = False
                        else:
                            raise TsFileParseException("invalid univariate value")

                        has_univariate_tag = True
                        metadata_started = True
                    elif line.startswith("@classlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("classlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            class_labels = True
                        elif tokens[1] == "false":
                            class_labels = False
                        else:
                            raise TsFileParseException("invalid classLabel value")

                        # Check if we have any associated class values
                        if token_len == 2 and class_labels:
                            raise TsFileParseException("if the classlabel tag is true then class values must be supplied")

                        has_class_labels_tag = True
                        class_label_list = [token.strip() for token in tokens[2:]]
                        metadata_started = True
                    elif line.startswith("@targetlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.split(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("targetlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            target_labels = True
                        elif tokens[1] == "false":
                            target_labels = False
                        else:
                            raise TsFileParseException("invalid targetLabel value")

                        has_target_labels_tag = True
                        class_val_list = []
                        metadata_started = True
                    # Check if this line contains the start of data
                    elif line.startswith("@data"):
                        if line != "@data":
                            raise TsFileParseException("data tag should not have an associated value")

                        if data_started and not metadata_started:
                            raise TsFileParseException("metadata must come before data")
                        else:
                            has_data_tag = True
                            data_started = True
                    # If the 'data tag has been found then metadata has been parsed and data can be loaded
                    elif data_started:
                        # Check that a full set of metadata has been provided
                        incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                        incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                        if incomplete_regression_meta_data and incomplete_classification_meta_data:
                            raise TsFileParseException("a full set of metadata has not been provided before the data")

                        # Replace any missing values with the value specified
                        line = line.replace("?", replace_missing_vals_with)

                        # Check if we dealing with data that has timestamps
                        if timestamps:
                            # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                            has_another_value = False
                            has_another_dimension = False

                            timestamps_for_dimension = []
                            values_for_dimension = []

                            this_line_num_dimensions = 0
                            line_len = len(line)
                            char_num = 0

                            while char_num < line_len:
                                # Move through any spaces
                                while char_num < line_len and str.isspace(line[char_num]):
                                    char_num += 1

                                # See if there is any more data to read in or if we should validate that read thus far

                                if char_num < line_len:

                                    # See if we have an empty dimension (i.e. no values)
                                    if line[char_num] == ":":
                                        if len(instance_list) < (this_line_num_dimensions + 1):
                                            instance_list.append([])

                                        instance_list[this_line_num_dimensions].append(pd.Series())
                                        this_line_num_dimensions += 1

                                        has_another_value = False
                                        has_another_dimension = True

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                        char_num += 1
                                    else:
                                        # Check if we have reached a class label
                                        if line[char_num] != "(" and target_labels:
                                            class_val = line[char_num:].strip()

                                            # if class_val not in class_val_list:
                                            #     raise TsFileParseException(
                                            #         "the class value '" + class_val + "' on line " + str(
                                            #             line_num + 1) + " is not valid")

                                            class_val_list.append(float(class_val))
                                            char_num = line_len

                                            has_another_value = False
                                            has_another_dimension = False

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                                        else:

                                            # Read in the data contained within the next tuple

                                            if line[char_num] != "(" and not target_labels:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not start with a '('")

                                            char_num += 1
                                            tuple_data = ""

                                            while char_num < line_len and line[char_num] != ")":
                                                tuple_data += line[char_num]
                                                char_num += 1

                                            if char_num >= line_len or line[char_num] != ")":
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not end with a ')'")

                                            # Read in any spaces immediately after the current tuple

                                            char_num += 1

                                            while char_num < line_len and str.isspace(line[char_num]):
                                                char_num += 1

                                            # Check if there is another value or dimension to process after this tuple

                                            if char_num >= line_len:
                                                has_another_value = False
                                                has_another_dimension = False

                                            elif line[char_num] == ",":
                                                has_another_value = True
                                                has_another_dimension = False

                                            elif line[char_num] == ":":
                                                has_another_value = False
                                                has_another_dimension = True

                                            char_num += 1

                                            # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                            last_comma_index = tuple_data.rfind(',')

                                            if last_comma_index == -1:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has no comma inside of it")

                                            try:
                                                value = tuple_data[last_comma_index + 1:]
                                                value = float(value)

                                            except ValueError:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                            # Check the type of timestamp that we have

                                            timestamp = tuple_data[0: last_comma_index]

                                            try:
                                                timestamp = int(timestamp)
                                                timestamp_is_int = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_int = False

                                            if not timestamp_is_int:
                                                try:
                                                    timestamp = float(timestamp)
                                                    timestamp_is_float = True
                                                    timestamp_is_timestamp = False
                                                except ValueError:
                                                    timestamp_is_float = False

                                            if not timestamp_is_int and not timestamp_is_float:
                                                try:
                                                    timestamp = timestamp.strip()
                                                    timestamp_is_timestamp = True
                                                except ValueError:
                                                    timestamp_is_timestamp = False

                                            # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                            if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                            if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                                raise TsFileParseException(
                                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            # Store the values

                                            timestamps_for_dimension += [timestamp]
                                            values_for_dimension += [value]

                                            #  If this was our first tuple then we store the type of timestamp we had

                                            if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                                previous_timestamp_was_timestamp = True
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_int is None and timestamp_is_int:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = True
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_float is None and timestamp_is_float:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = True

                                            # See if we should add the data for this dimension

                                            if not has_another_value:
                                                if len(instance_list) < (this_line_num_dimensions + 1):
                                                    instance_list.append([])

                                                if timestamp_is_timestamp:
                                                    timestamps_for_dimension = pd.DatetimeIndex(timestamps_for_dimension)

                                                instance_list[this_line_num_dimensions].append(
                                                    pd.Series(index=timestamps_for_dimension, data=values_for_dimension))
                                                this_line_num_dimensions += 1

                                                timestamps_for_dimension = []
                                                values_for_dimension = []

                                elif has_another_value:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ',' that is not followed by another tuple")

                                elif has_another_dimension and target_labels:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ':' while it should list a class value")

                                elif has_another_dimension and not target_labels:
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(pd.Series(dtype=np.float32))
                                    this_line_num_dimensions += 1
                                    num_dimensions = this_line_num_dimensions

                                # If this is the 1st line of data we have seen then note the dimensions

                                if not has_another_value and not has_another_dimension:
                                    if num_dimensions is None:
                                        num_dimensions = this_line_num_dimensions

                                    if num_dimensions != this_line_num_dimensions:
                                        raise TsFileParseException("line " + str(
                                            line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check that we are not expecting some more data, and if not, store that processed above

                            if has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(pd.Series())
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and num_dimensions != this_line_num_dimensions:
                                raise TsFileParseException("line " + str(
                                    line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check if we should have class values, and if so that they are contained in those listed in the metadata

                            if target_labels and len(class_val_list) == 0:
                                raise TsFileParseException("the cases have no associated class values")
                        else:
                            dimensions = line.split(":")
                            # If first row then note the number of dimensions (that must be the same for all cases)
                            if is_first_case:
                                num_dimensions = len(dimensions)

                                if target_labels:
                                    num_dimensions -= 1

                                for dim in range(0, num_dimensions):
                                    instance_list.append([])
                                is_first_case = False

                            # See how many dimensions that the case whose data in represented in this line has
                            this_line_num_dimensions = len(dimensions)

                            if target_labels:
                                this_line_num_dimensions -= 1

                            # All dimensions should be included for all series, even if they are empty
                            if this_line_num_dimensions != num_dimensions:
                                raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                    num_dimensions) + " but have read " + str(this_line_num_dimensions))

                            # Process the data for each dimension
                            for dim in range(0, num_dimensions):
                                dimension = dimensions[dim].strip()

                                if dimension:
                                    data_series = dimension.split(",")
                                    data_series = [float(i) for i in data_series]
                                    instance_list[dim].append(pd.Series(data_series))
                                else:
                                    instance_list[dim].append(pd.Series())

                            if target_labels:
                                class_val_list.append(float(dimensions[num_dimensions].strip()))

                line_num += 1

        # Check that the file was not empty
        if line_num:
            # Check that the file contained both metadata and data
            complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
            complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

            if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
                raise TsFileParseException("metadata incomplete")
            elif metadata_started and not data_started:
                raise TsFileParseException("file contained metadata but no data")
            elif metadata_started and data_started and len(instance_list) == 0:
                raise TsFileParseException("file contained metadata but no data")

            # Create a DataFrame from the data parsed above
            data = pd.DataFrame(dtype=np.float32)

            for dim in range(0, num_dimensions):
                data['dim_' + str(dim)] = instance_list[dim]

            # Check if we should return any associated class labels separately

            if target_labels:
                if return_separate_X_and_y:
                    return data, np.asarray(class_val_list)
                else:
                    data['class_vals'] = pd.Series(class_val_list)
                    return data
            else:
                return data
        else:
            raise TsFileParseException("empty file")

    def download(self, raw_data_dir):
        ## Unfortunately, all Monash, UEA & UCR Time Series Regression datasets have to be downloaded, although we only need the HR prediction data set
        url = 'https://zenodo.org/record/3902651/files/Monash_UEA_UCR_Regression_Archive.zip?download=1'
        save_dir = raw_data_dir
        zipname = save_dir + '/uea_reg.zip'
        ## download zipped data
        urllib.request.urlretrieve(url, zipname)
        ## unzip:
        with zipfile.ZipFile(zipname, 'r') as zip:
            zip.extractall(save_dir)

    def process_data(self, raw_data_dir, processed_data_dir):
        rnd_state = 123456
        train_ts = raw_data_dir + '/BIDMC32HR/BIDMC32HR_TRAIN.ts'
        test_ts = raw_data_dir + '/BIDMC32HR/BIDMC32HR_TEST.ts'
        X_train, y_train = self.load_from_tsfile_to_dataframe(train_ts)
        X_test, y_test = self.load_from_tsfile_to_dataframe(test_ts)
        all_frames = pd.concat((X_train, X_test))

        ## Bit taken from https://github.com/jambo6/neuralRDEs/blob/master/get_data/tsr.py
        tensor_labels = torch.Tensor(np.concatenate((y_train, y_test)))
        tensor_data = []
        for idx in range(all_frames.shape[0]):
            tensor_data.append(torch.Tensor(pd.concat(all_frames.iloc[idx].values, axis=1).values))
        tensor_data = torch.stack(tensor_data)
        dataset = TensorDataset(tensor_data, tensor_labels)

        train_dataset, valid_test_dataset = torch.utils.data.random_split(dataset, [5565, 2384], generator=torch.Generator().manual_seed(123456))
        test_dataset, valid_dataset = torch.utils.data.random_split(valid_test_dataset, [1192, 1192], generator=torch.Generator().manual_seed(123456))
        torch.save(train_dataset, processed_data_dir + '/training.pt')
        torch.save(test_dataset, processed_data_dir + '/test.pt')
        torch.save(valid_dataset, processed_data_dir + '/validation.pt')

    def get_heart_rate(self):
        data_dir = os.getcwd()  + '/data/heart_rate'
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        raw_data_dir = data_dir + '/raw'
        if os.path.isdir(raw_data_dir):
            print("Data already downloaded")
        else:
            os.mkdir(raw_data_dir)
            print("Downloading data")
            self.download(raw_data_dir)
            print("Data download finished")

        processed_data_dir = data_dir + '/processed'
        if os.path.isdir(processed_data_dir):
            print("Data already processed")
        else:
            os.mkdir(processed_data_dir)
            self.process_data(raw_data_dir, processed_data_dir)
            print("Finished processing data")

        train_dataset = torch.load(processed_data_dir + '/training.pt')
        test_dataset = torch.load(processed_data_dir + '/test.pt')
        valid_dataset = torch.load(processed_data_dir + '/validation.pt')

        return train_dataset, test_dataset, valid_dataset
    