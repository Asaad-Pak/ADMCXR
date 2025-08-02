import pandas as pd
from ADMCXR_dataset import CXRTestDataset_h5
from ADMCXR_config import ADMCXRConfig

dset_cosine =  CXRTestDataset_h5(r"F:\\Asaad_Output\\Preprocessing\\bootstrap_test\\cxr.h5", 256)
dset_itm =  CXRTestDataset_h5(r"F:\\Asaad_Output\\Preprocessing\\bootstrap_test\\cxr.h5", 384)
train_data = pd.read_csv(r"F:\\Asaad_Output\\Preprocessing\\mimic_train_impressions.csv")
reports = train_data["report"].drop_duplicates().dropna().reset_index(drop = True)

config = ADMCXRConfig(
        albef_retrieval_config = r"F:\\Asaad_XREM_Repo\\ADMCXR\\configs\\Cosine-Retrieval.yaml",
        albef_retrieval_ckpt = r"F:\\Asaad_Output\\Pretraining\\checkpoint_58.pth",
        albef_itm_config = r"F:\\Asaad_XREM_Repo\\ADMCXR\\configs\\ITM.yaml",
        albef_itm_ckpt = r"F:\\Asaad_Output\\ITM_TR_150K\\checkpoint_6.pth",
)

from ADMCXR import ADMCXR
admcxr = ADMCXR(config)
print("XREM MODEL: ",admcxr)

print("Starting Inference")
itm_output = admcxr(reports, dset_cosine, dset_itm)
print('Inference Done Now Saving Result')
itm_df = pd.DataFrame(itm_output, columns=['Report Impression'])
itm_df.to_csv(r"F:\\Asaad_Output\\Inf_3678\\E_RV\\init_result_E_RV.csv", index=False)