import pandas as pd
import numpy as np


if __name__=='__main__':
    train = pd.read_csv('cancer_detection/data/train_positive.csv')
    target_path = 'cancer_detection/data/train_positive1.csv'
    idx = 0
    header = True

    for i in range(len(train)):
        print('\rtrain_positive Document Generating ProgressEval: {}/{}'.format(i, len(train)), end="")
        data_frame = pd.DataFrame({'site_id': train.loc[i]['site_id'],
                                    'patient_id': train.loc[i]['patient_id'],
                                    'image_id': train.loc[i]['image_id'],
                                    'laterality': train.loc[i]['laterality'],
                                    'view': train.loc[i]['view'],
                                    'age': train.loc[i]['age'],
                                    'cancer': train.loc[i]['cancer'],
                                    'biopsy': train.loc[i]['biopsy'],
                                    'invasive': train.loc[i]['invasive'],
                                    'BIRADS': train.loc[i]['BIRADS'],
                                    'implant': train.loc[i]['implant'],
                                    'density': train.loc[i]['density'],
                                    'machine_id': train.loc[i]['machine_id'],
                                    'difficult_negative_case': train.loc[i]['difficult_negative_case']}, index=[idx])
        if idx != 0:
            header = False
        idx += 1
        data_frame.to_csv(target_path, mode='a', header=header, index=False)

        data_frame = pd.DataFrame({'site_id': train.loc[i]['site_id'],
                                    'patient_id': train.loc[i]['patient_id'],
                                    'image_id': str(train.loc[i]['image_id']) + '0',
                                    'laterality': train.loc[i]['laterality'],
                                    'view': train.loc[i]['view'],
                                    'age': train.loc[i]['age'],
                                    'cancer': train.loc[i]['cancer'],
                                    'biopsy': train.loc[i]['biopsy'],
                                    'invasive': train.loc[i]['invasive'],
                                    'BIRADS': train.loc[i]['BIRADS'],
                                    'implant': train.loc[i]['implant'],
                                    'density': train.loc[i]['density'],
                                    'machine_id': train.loc[i]['machine_id'],
                                    'difficult_negative_case': train.loc[i]['difficult_negative_case']}, index=[idx])
        if idx != 0:
            header = False
        idx += 1
        data_frame.to_csv(target_path, mode='a', header=header, index=False)

        data_frame = pd.DataFrame({'site_id': train.loc[i]['site_id'],
                                    'patient_id': train.loc[i]['patient_id'],
                                    'image_id': str(train.loc[i]['image_id']) + '1',
                                    'laterality': train.loc[i]['laterality'],
                                    'view': train.loc[i]['view'],
                                    'age': train.loc[i]['age'],
                                    'cancer': train.loc[i]['cancer'],
                                    'biopsy': train.loc[i]['biopsy'],
                                    'invasive': train.loc[i]['invasive'],
                                    'BIRADS': train.loc[i]['BIRADS'],
                                    'implant': train.loc[i]['implant'],
                                    'density': train.loc[i]['density'],
                                    'machine_id': train.loc[i]['machine_id'],
                                    'difficult_negative_case': train.loc[i]['difficult_negative_case']}, index=[idx])
        if idx != 0:
            header = False
        idx += 1
        data_frame.to_csv(target_path, mode='a', header=header, index=False)