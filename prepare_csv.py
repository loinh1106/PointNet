import pandas as pd
import numpy as np
from glob import glob

def prepare_csv(root,csv_path, output_path):
    df = pd.read_csv(csv_path)
    lst_class = glob(f"{root}/ModelNet10/*")
    lst_class.sort()
    id2id_encode = {x.split("/")[-1]:idx for idx, x in enumerate(lst_class)}
    df['class_id'] = df['class'].map(id2id_encode)
    class_id2idx = {class_id: idx for idx, class_id in enumerate(sorted(df['class_id'].unique()))}
    df['id_encode'] = df['class_id'].map(class_id2idx)
    train_df = df[df['split']== 'train']
    test_df = df[df['split']=='test']
    train_df.to_csv(f'{output_path}/train.csv', index=False)
    test_df.to_csv(f'{output_path}/test.csv', index=False)
    print('Done!')
    

def main():
    prepare_csv('./data','./data/metadata_modelnet10.csv', './data')

if __name__ =='__main__':
    main()