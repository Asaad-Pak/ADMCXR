import pandas as pd
import argparse

#CXR-Metric computes the scores based on 2,192 samples from the mimic-cxr test set that consists of 3,678
#This function selects the respective 2,192 reports 
def main(fpath, opath):
    df = pd.read_csv(fpath)
    test = pd.read_csv('/mnt/f/Asaad_Output/Inf_A/labels.csv')
    gt = pd.read_csv('/mnt/f/Asaad_Output/Inf_A/reports.csv')[['dicom_id', 'study_id']] #contains the dicom ids of the 2,192 imgs
    gt['report'] = [None] * len(gt)
    pred = df['filtered']
    pred = pd.concat([test[['dicom_id']], pred], axis = 1)
    pred = pred.set_index('dicom_id')
    for idx, row in gt.iterrows():
        dicom_id, study_id, _ = row
        # gt['report'][idx] = pred['filtered'][dicom_id]
        gt.loc[idx, 'report'] = pred.loc[dicom_id, 'filtered']
    gt.to_csv(opath, index = False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='')
    parser.add_argument('--opath', default='predict.csv') 
    args = parser.parse_args()
    main(args.fpath, args.opath)