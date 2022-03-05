from posixpath import split
import pandas as pd
import csv
import os
import numpy as np

def convert_local(dataFrame, path):

    if not os.path.exists(path):
        os.mkdir(path)
    
    df = dataFrame[['headline','text']].copy()
    df["headline"] = df["headline"].str[1:]
    df["text"] = df["text"].str.replace("\n"," ",regex=False).str.strip().str.replace('\,','', regex=True).str.replace(r' +', ' ', regex=True)
    
    for index, _ in df.iterrows():
        df.iloc[index].to_csv(os.path.join(path, f'{index}.txt'), index=False, header=False,quoting=csv.QUOTE_NONE,escapechar="\\")

def split_data(data_path):

    data = pd.read_csv(data_path)
    train_ind = []
    val_ind = []
    test_ind = []

    with open('all_train.txt', 'r', encoding='UTF-8') as f:
        train_titles = set(f.read().splitlines())

    with open('all_val.txt', 'r', encoding='UTF-8') as f:
        val_titles = set(f.read().splitlines())

    with open('all_test.txt', 'r', encoding='UTF-8') as f:
        test_titles = set(f.read().splitlines())

    for ind, row in data.iterrows():

        if str(row['title']).replace(' ','').replace('\n','') in train_titles:
            if pd.isnull(row.headline) or pd.isnull(row.text) or row.text.replace('\n','').replace(' ','') == '' or row.headline.replace('\n','').replace(' ','') == '':
                continue
            train_ind.append(ind)

        if str(row['title']).replace(' ','').replace('\n','') in val_titles:
            if pd.isnull(row.headline) or pd.isnull(row.text) or row.text.replace('\n','').replace(' ','') == '' or row.headline.replace('\n','').replace(' ','') == '':
                continue
            val_ind.append(ind)

        if str(row['title']).replace(' ','').replace('\n','') in test_titles:
            if pd.isnull(row.headline) or pd.isnull(row.text) or row.text.replace('\n','').replace(' ','') == '' or row.headline.replace('\n','').replace(' ','') == '':
                continue
            test_ind.append(ind)

    return data.iloc[train_ind].copy().reset_index(drop=True), data.iloc[val_ind].copy().reset_index(drop=True), data.iloc[test_ind].copy().reset_index(drop=True)

if __name__ == '__main__':

    import time

    start = time.time()

    train, val, test = split_data("wikihowSep.csv")
    convert_local(train, 'train')
    convert_local(val, 'val')
    convert_local(test, 'test')

    end = time.time()

    print(f'Conversion has been done in {end-start:.2f} seconds')