import numpy as np
import pandas as pd
import argparse


def inverse_transform(data, new_cols):
    
    
    def extract(vals, cols):
        
        index= np.argmax(vals)
        
        return cols[index]
        
        
    df = data.copy()
            
    step_size = 3
    start = 0
    
    for i, cols_j in enumerate(range(start, len(new_cols), step_size)):
        
        start = cols_j
        
        cols_i = new_cols[start: (start + step_size)]
        
        print(f'Columns idexed from {start} to {start + step_size} --> {cols_i}')
        
        df.loc[:, 'temp_col_'+str(i)] = df[cols_i].apply(lambda s : extract(s.values, cols_i), axis = 1)
                
        col_name = '_'.join(cols_i[0].split('_')[:-1])
                
        df.loc[:, col_name] = df['temp_col_'+str(i)].apply(lambda k : k.split('_')[-1])
        
    return df

def transform_c_hdl(row):
    return str(row["Reading_ID"]) + "_hdl_cholesterol_human" + "-" +  row["hdl_cholesterol_human"]
def transform_hemo(row):
    return str(row["Reading_ID"]) + "_hemoglobin(hgb)_human" +  "-" + row["hemoglobin(hgb)_human"]
def transform_c_ldl(row):
    return str(row["Reading_ID"]) + "_cholesterol_ldl_human" +  "-" + row["cholesterol_ldl_human"]


def main(args):
    
    data = pd.read_csv(f'data/test_{args.model_name}.csv')
        
    targets = ['hdl_cholesterol_human_ok','hdl_cholesterol_human_high', 'hdl_cholesterol_human_low', 
                'cholesterol_ldl_human_ok', 'cholesterol_ldl_human_high', 'cholesterol_ldl_human_low',
               'hemoglobin(hgb)_human_ok', 'hemoglobin(hgb)_human_high', 'hemoglobin(hgb)_human_low'
               ]
    
    o_targets = ['hdl_cholesterol_human', 'cholesterol_ldl_human', 'hemoglobin(hgb)_human']
    
    predictions_ = data[targets].values
    
    preds = (predictions_ > 0.5).astype(int)
    
    data.loc[:, targets] = preds
    
    test_ = inverse_transform(data, targets)
    
    hdl_rows = pd.DataFrame(test_[['Reading_ID'] + o_targets].apply(transform_c_hdl, axis=1))
    hemo_rows = pd.DataFrame(test_[['Reading_ID'] + o_targets].apply(transform_hemo, axis=1))
    ldl_rows = pd.DataFrame(test_[['Reading_ID'] + o_targets].apply(transform_c_ldl, axis=1))
    
    ss = pd.concat([hdl_rows, hemo_rows, ldl_rows]).reset_index(drop=True)    
    ss["target"] = ss[0].apply(lambda x: x.split("-")[1])
    ss[0] = ss[0].apply(lambda x: x.split("-")[0])
    
    ss = ss.rename(columns={0:"Reading_ID"})
    
    ss.to_csv(f'{args.submission_dir}/zindi_submission_{args.model_name}.csv', index=False)
    print('Submission file successfully created!')
    
    
if __name__ =="__main__":
    
    parser = argparse.ArgumentParser(description = "create zindi submission file from the model ouptut")
    parser.add_argument("--model_name", type = str, default = None, help ="name of model output file")
    parser.add_argument("--submission_dir", type = str, default = 'submissions', help ="path/to/submission directory")
    
    args = parser.parse_args()
        
    main(args)

    
    
    