import pandas as pd

if __name__ == '__main__':
    df1 = pd.read_csv("cancer_detection/data/train_negative.csv")  # 读取第一个文件
    df2 = pd.read_csv("cancer_detection/data/train_positive1.csv")  # 读取第二个文件
    target_path = 'cancer_detection/data/train1.csv'
    idx = 0
    header = True

    for i in range(len(df1)):
        print('\rtrain_negative Document Generating ProgressEval: {}/{}'.format(i, len(df1)), end="")
        data_frame = pd.DataFrame({'site_id': df1.loc[i]['site_id'],
                                   'patient_id': df1.loc[i]['patient_id'],
                                   'image_id': df1.loc[i]['image_id'],
                                   'laterality': df1.loc[i]['laterality'],
                                   'view': df1.loc[i]['view'],
                                   'age': df1.loc[i]['age'],
                                   'cancer': df1.loc[i]['cancer'],
                                   'biopsy': df1.loc[i]['biopsy'],
                                   'invasive': df1.loc[i]['invasive'],
                                   'BIRADS': df1.loc[i]['BIRADS'],
                                   'implant': df1.loc[i]['implant'],
                                   'density': df1.loc[i]['density'],
                                   'machine_id': df1.loc[i]['machine_id'],
                                   'difficult_negative_case': df1.loc[i]['difficult_negative_case']}, index=[idx])
        if idx != 0:
            header = False
        idx += 1
        data_frame.to_csv(target_path, mode='a', header=header, index=False)

    for i in range(len(df2)):
        print('\rtrain_positive1 Document Generating ProgressEval: {}/{}'.format(i, len(df2)), end="")
        data_frame = pd.DataFrame({'site_id': df2.loc[i]['site_id'],
                                   'patient_id': df2.loc[i]['patient_id'],
                                   'image_id': df2.loc[i]['image_id'],
                                   'laterality': df2.loc[i]['laterality'],
                                   'view': df2.loc[i]['view'],
                                   'age': df2.loc[i]['age'],
                                   'cancer': df2.loc[i]['cancer'],
                                   'biopsy': df2.loc[i]['biopsy'],
                                   'invasive': df2.loc[i]['invasive'],
                                   'BIRADS': df2.loc[i]['BIRADS'],
                                   'implant': df2.loc[i]['implant'],
                                   'density': df2.loc[i]['density'],
                                   'machine_id': df2.loc[i]['machine_id'],
                                   'difficult_negative_case': df2.loc[i]['difficult_negative_case']}, index=[idx])
        if idx != 0:
            header = False
        idx += 1
        data_frame.to_csv(target_path, mode='a', header=header, index=False)