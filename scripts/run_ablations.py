import argparse
import numpy as np

from methods.DSM import KDSM, DSM, KDSM_GGD, KDSM_EMATeacher, DSM_EMATeacher

import os
import pandas as pd

import time

from adbench.myutils import Utils

import sklearn.metrics as skm
from utils.data_generator import DataGenerator

def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies-1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds

def main(args):
    seed = args.seed
    setting = args.setting
    
    dir = './ablations/' + setting + '/'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    if "semi" in setting:
        datagenerator = DataGenerator(seed = seed, test_size=0.5, normal=True) # data generator
    else:
        datagenerator = DataGenerator(seed = seed, test_size=0, normal=False) # data generator
    
    utils = Utils() # utils function
    utils.set_seed(seed)
    
    model_dict = {}
    
    if args.setting == "unsup":
        model_dict['K-DSM-EMA_75'] = lambda seed, model_name: KDSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=75)
        model_dict['K-DSM-EMA_80'] = lambda seed, model_name: KDSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=80)
        model_dict['K-DSM-EMA_85'] = lambda seed, model_name: KDSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=85)
        model_dict['K-DSM-EMA_90'] = lambda seed, model_name: KDSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=90)
        model_dict['K-DSM-EMA_95'] = lambda seed, model_name: KDSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=95)
        
        model_dict['DSM-EMA_75'] = lambda seed, model_name: DSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=75)
        model_dict['DSM-EMA_80'] = lambda seed, model_name: DSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=80)
        model_dict['DSM-EMA_85'] = lambda seed, model_name: DSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=85)
        model_dict['DSM-EMA_90'] = lambda seed, model_name: DSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=90)
        model_dict['DSM-EMA_95'] = lambda seed, model_name: DSM_EMATeacher(seed=seed, model_name=model_name, filter_pct=95)
    else:
        model_dict['K-DSM_2_0.1'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, sigma=0.1, loss_type="squared")
        model_dict['K-DSM_2_0.3'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, sigma=0.3, loss_type="squared")
        model_dict['K-DSM_2_0.5'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, sigma=0.5, loss_type="squared")
        model_dict['K-DSM_2_0.7'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, sigma=0.7, loss_type="squared")
        model_dict['K-DSM_2_1.0'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, sigma=1.0, loss_type="squared")

        model_dict['K-DSMc_0.875_0.00001'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.00001)
        model_dict['K-DSMc_0.560_0.00005'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.00005)
        model_dict['K-DSMc_0.506_0.0001'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.0001)
        model_dict['K-DSMc_0.326_0.001'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.001)
        model_dict['K-DSMc_0.203_0.005'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.005)
        model_dict['K-DSMc_0.151_0.01'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.01)
        model_dict['K-DSMc_0.035_0.05'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.05)
        model_dict['K-DSMc_0.012_0.07'] = lambda seed, model_name: KDSM(seed=seed, model_name=model_name, tau=0.07)
        
        model_dict['K-DSMggd_0.000005'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.000005)
        model_dict['K-DSMggd_0.00001'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.00001)
        model_dict['K-DSMggd_0.00005'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.00005)
        model_dict['K-DSMggd_0.0001'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.0001)
        model_dict['K-DSMggd_0.001'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.001)
        model_dict['K-DSMggd_0.005'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.005)
        model_dict['K-DSMggd_0.01'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.01)
        model_dict['K-DSMggd_0.05'] = lambda seed, model_name: KDSM_GGD(seed=seed, model_name=model_name, tau=0.05)
        
        model_dict['DSM_0.1'] = lambda seed, model_name: DSM(seed=seed, model_name=model_name, sigma=0.1)
        model_dict['DSM_0.3'] = lambda seed, model_name: DSM(seed=seed, model_name=model_name, sigma=0.3)
        model_dict['DSM_0.5'] = lambda seed, model_name: DSM(seed=seed, model_name=model_name, sigma=0.5)
        model_dict['DSM_0.7'] = lambda seed, model_name: DSM(seed=seed, model_name=model_name, sigma=0.7)
        model_dict['DSM_1.0'] = lambda seed, model_name: DSM(seed=seed, model_name=model_name, sigma=1.0)
    
    # Create dataframes to save the results
    aucroc_name = dir + str(seed) + "_AUCROC.csv"
    aucpr_name = dir + str(seed) + "_AUCPR.csv"
    f1_name = dir + str(seed) + "_AUCF1.csv"
    train_name = dir + str(seed) + "_TrainTime.csv"
    inference_name = dir + str(seed) + "_InferenceTime.csv"
    
    try:
        df_AUCROC = pd.read_csv(aucroc_name, index_col = 0) 
    except:
        df_AUCROC = pd.DataFrame(data=None)
    try:
        df_AUCPR = pd.read_csv(aucpr_name, index_col = 0)
    except:
        df_AUCPR = pd.DataFrame(data=None)
    try:
        df_F1 = pd.read_csv(f1_name, index_col = 0)
    except:
        df_F1 = pd.DataFrame(data=None)
    try:
        df_train = pd.read_csv(train_name, index_col = 0)
    except:
        df_train = pd.DataFrame(data=None)
    try:
        df_inference = pd.read_csv(inference_name, index_col = 0)
    except:
        df_inference = pd.DataFrame(data=None)
    
    # Get the datasets from ADBench
    for dataset_list in [datagenerator.dataset_list_classical, datagenerator.dataset_list_cv, datagenerator.dataset_list_nlp]:
        for dataset in dataset_list:
            '''
            la: ratio of labeled anomalies, from 0.0 to 1.0
            realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
            noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
            '''
            print(dataset)

            # import the dataset
            datagenerator.dataset = dataset # specify the dataset name
            data = datagenerator.generator(la=0, max_size=50000) # maximum of 50,000 data points are available
            
            if "unsup" in setting:
                X = data['X_test']
                y = data['y_test']
                
                indices = np.arange(len(X))
                subset = np.random.choice(indices, size = len(indices), replace=True)
                
                data = {}
                data['X_train'] = X[subset]
                data['y_train'] = y[subset]
                
                data['X_test'] = X
                data['y_test'] = y
            
            for name, clf in model_dict.items():
                # Skip if already computed, unless --override is set
                if not args.override:
                    if (dataset in df_AUCPR.index and name in df_AUCPR.columns
                            and not pd.isna(df_AUCPR.loc[dataset, name])):
                        print(f"Skipping {name} on {dataset} (already computed)")
                        continue

                # model initialization
                print(name)
                clf = clf(seed=seed, model_name=name)
                
                # training, for unsupervised models the y label will be discarded
                start_time = time.time()
                clf = clf.fit(data['X_train'], np.zeros_like(data['y_train']))
                end_time = time.time(); time_fit = end_time - start_time 
                
                start_time = time.time()
                if name == 'DAGMM':
                    score = clf.predict_score(data['X_train'], data['X_test'])
                else:
                    score = clf.predict_score(data['X_test'])
                end_time = time.time(); time_inference = end_time - start_time
                
                indices = np.arange(len(data['y_test']))
                p = low_density_anomalies(-score, len(indices[data['y_test']==1]))
                f1_score = skm.f1_score(data['y_test'], p)
                print('F1 score: ' + str(f1_score))

                df_F1.loc[dataset, name] = f1_score
                df_F1.to_csv(f1_name)

                inds = np.where(np.isnan(score))
                score[inds] = 0
                
                result = utils.metric(y_true=data['y_test'], y_score=score)
                print('AUCROC: ' + str(result['aucroc']))
                
                # save results
                df_AUCROC.loc[dataset, name] = result['aucroc']
                df_AUCPR.loc[dataset, name] = result['aucpr']
                
                df_train.loc[dataset, name] = time_fit
                df_inference.loc[dataset, name] = time_inference
                
                df_AUCROC.to_csv(aucroc_name)
                df_AUCPR.to_csv(aucpr_name)
                
                df_train.to_csv(train_name)
                df_train.to_csv(train_name)
                
                df_inference.to_csv(inference_name)
                df_inference.to_csv(inference_name)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--setting', type=str,
        default='semi', help='choice of experimental setting (semi or unsup)')
    parser.add_argument('--seed', type=int, 
        default=42, help='random seed')
    parser.add_argument('--override', action='store_true',
        help='re-run all experiments, even if results already exist')

    args = parser.parse_args()
    main(args)
