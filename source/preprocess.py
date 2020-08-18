import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


def preprocess_data(train_df, test_df, submission_df):
    """
    Pre-processing of the train and test dataset: it creates pp_train.csv and pp_test.csv for training and inference
    :param train_df:
    :param test_df:
    :param submission_df:
    :return:
    """
    train_c_df = train_df.copy()
    test_c_df = test_df.copy()
    train_c_df['mode'] = 1  # train
    test_c_df['mode'] = 0  # test
    submission_df['mode'] = 0  # test

    # add submission data to have more rows to test
    submission_df['Patient'] = submission_df['Patient_Week'].apply(lambda x: x.split('_')[0])
    submission_df['Weeks'] = submission_df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    submission_df = submission_df[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
    submission_df = submission_df.merge(test_c_df.drop('Weeks', axis=1), on="Patient")

    data_df = train_c_df.append([test_c_df, submission_df])

    data_df['min_week'] = data_df['Weeks']

    data_df.loc[data_df['mode'] == 0, 'min_week'] = np.nan
    data_df['min_week'] = data_df.groupby('Patient')['min_week'].transform('min')

    base_visit = data_df.loc[data_df.Weeks == data_df.min_week]  # take records with min week.
    base_visit = base_visit[['Patient', 'FVC']].copy()  # copy patient and fvc
    base_visit.columns = ['Patient', 'min_FVC'] # rename fvc to min_fvc as corresponds to min week
    base_visit['nb'] = 1
    base_visit['nb'] = base_visit.groupby('Patient')['nb'].transform('cumsum') # counts num of patients wit
    base_visit = base_visit[base_visit.nb == 1] # df with the base week for all the patients
    base_visit.drop('nb', axis=1, inplace=True)

    data_df = data_df.merge(base_visit, on='Patient', how='left') # merge data with base: ie add min_FVC column
    data_df['weeks_from_min'] = data_df['Weeks'] - data_df['min_week'] # difference

    # normalise sex and smoking status in single columns per value
    COLS = ['Sex', 'SmokingStatus']
    for col in COLS:
        for mod in data_df[col].unique():
            data_df[mod] = (data_df[col] == mod).astype(int)

    results_df = pd.DataFrame({'Patient': data_df[data_df['mode'] == 0]['Patient'],
                              'Weeks': data_df[data_df['mode'] == 0]['Weeks']})
    results_df = results_df.reset_index(drop=True)

    data_df = data_df.drop(columns=['Patient', 'FVC', 'min_week', 'Sex', 'SmokingStatus', 'Confidence', 'Patient_Week'])

    # normilise dataframe
    min_max_scaler = preprocessing.MinMaxScaler()
    data_df_scaled = pd.DataFrame(min_max_scaler.fit_transform(data_df), columns=data_df.columns)
    data_df_scaled.astype({'mode': 'int32'})

    #split train and test to normalise
    pp_train_df = data_df_scaled[data_df_scaled['mode'] == 1]
    pp_test_df = data_df_scaled[data_df_scaled['mode'] == 0]
    pp_train_df = pp_train_df.drop(columns=['mode'])
    pp_test_df = pp_test_df.drop(columns=['mode'])

    #puts FVC as the first columns for both pp_train_df and pp_test_df
    pp_test_df = pp_test_df.reset_index(drop=True)
    pp_train_df['FVC'] = train_df['FVC']
    pp_test_df['FVC'] = test_df['FVC']

    pp_train_cols = list(pp_train_df.columns)
    pp_train_cols[0], pp_train_cols[len(pp_train_cols)-1] = pp_train_cols[len(pp_train_cols)-1], pp_train_cols[0]
    pp_train_df = pp_train_df[pp_train_cols]

    pp_test_cols = list(pp_test_df.columns)
    pp_test_cols[0], pp_test_cols[len(pp_test_cols)-1] = pp_test_cols[len(pp_test_cols)-1], pp_test_cols[0]
    pp_test_df = pp_test_df[pp_test_cols]

    return pp_train_df, pp_test_df, results_df


if __name__ == '__main__':
    train_df = pd.read_csv(
        filepath_or_buffer=os.path.join("data/", "train_df.csv"),
        header=0, names=None)
    test_df = pd.read_csv(
        filepath_or_buffer=os.path.join("data/", "test_df.csv"),
        header=0, names=None)
    sub_df = pd.read_csv(filepath_or_buffer=os.path.join("data/", "sample_submission.csv"),
                                header=0, names=None)
    preprocess_data(train_df, test_df, sub_df)
