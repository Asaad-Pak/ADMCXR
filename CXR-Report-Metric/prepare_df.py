import pandas as pd
import argparse

#CXR-Metric computes the scores based on 2,192 samples from the mimic-cxr test set that consists of 3,678
#This function selects the respective 2,192 reports 
def main(fpath, opath):
    df = pd.read_csv(fpath)
    # test = pd.read_csv('F:\\Asaad_Output\\Preprocessing\\mimic_test_impressions.csv')
    # test = pd.read_csv('F:\\Asaad_Output\\Inference\\filtered_mimic_test_impressions.csv') # file with empty rows discarded
    # test = pd.read_csv('F:\\Asaad_Output\\Inference\\reports.csv') # file with empty rows discarded from bootstrap test #latest
    # test = pd.read_csv('F:\\Asaad_Output\\Inf_BCDM_3_USD_B\\reports.csv') #for Inf_BCDM_5_USD_B ablation best case on 2192 unique study
    test = pd.read_csv('F:\\Asaad_Output\\Inf_2192\\BCDM_3_USD_RV\\Sample\\reports.csv') # for sample collection from BCDM_3_USD_RV
    # test = pd.read_csv('F:\\Asaad_Output\\Inf_A\\reports.csv') # for order reversal test
    # gt = pd.read_csv('F:\\Asaad_Output\\Inference\\labels.csv')[['dicom_id', 'study_id']] #contains the dicom ids of the 2,192 imgs
    # gt = pd.read_csv('F:\\Asaad_Output\\Inference\\reports.csv')[['dicom_id', 'study_id']] #contains the dicom ids of the 2,192 imgs #pred 6
    # gt = pd.read_csv('F:\\Asaad_Output\\Inference\\reports_unique_study_id.csv')[['dicom_id', 'study_id']] #contains the dicom ids of the 2,192 imgs #pred 7
    # gt = pd.read_csv('F:\\Asaad_Output\\Inference\\reports_unique_study_id.csv')[['dicom_id', 'study_id']] #contains the dicom ids of the 2,192 imgs #pred 8
    # gt = pd.read_csv('F:\\Asaad_Output\\Inf_BCDM_3_USD_B\\reports_unique_study_id.csv')[['dicom_id', 'study_id']] #for Inf_BCDM_5_USD_B ablation best case on 2192 unique study
    gt = pd.read_csv('F:\\Asaad_Output\\Inf_2192\\BCDM_3_USD_RV\\Sample\\reports_unique_study_id.csv')[['dicom_id', 'study_id']] # for sample collection from BCDM_3_USD_RV
    # gt = pd.read_csv('F:\\Asaad_Output\\Inf_A\\labels.csv')[['dicom_id', 'study_id']] # for order reversal test
    gt['report'] = [None] * len(gt)
    pred = df['filtered']
    pred = pd.concat([test[['dicom_id']], pred], axis = 1)
    pred = pred.set_index('dicom_id')
    for idx, row in gt.iterrows():
        dicom_id, study_id, _ = row
        # gt['report'][idx] = pred['filtered'][dicom_id] #pred1 way 1
        gt.loc[idx, 'report'] = pred.loc[dicom_id, 'filtered'] #pred2 way 2
    gt.to_csv(opath, index = False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='')
    parser.add_argument('--opath', default='predict_E_RV_2.csv') 
    args = parser.parse_args()
    main(args.fpath, args.opath)