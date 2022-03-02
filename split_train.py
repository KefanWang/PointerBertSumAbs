import pandas as pd
path='wikihowSep.csv'
data=pd.read_csv(path)
title_fromcsv=data['title']
result=[]
with open("all_train.txt",'r',encoding='UTF-8') as f:
    for line_train in f:
        result.append(list(line_train.strip('\n').split(",")))
print(result[0])
print(result[1])
output_train=[]

for i in result:
    for j in range(len(title_fromcsv)):
        output=[]
        if type(title_fromcsv[j])==str:
            if i==title_fromcsv[j].replace(" ",""):
                output.append(data.iloc[j]["overview"])
                output.append(data.iloc[j]["headline"])
                output.append(data.iloc[j]["text"])
                output.append(data.iloc[j]["sectionLabel"])
                
                output_train.append(output)

with open("outputtrain.txt",'w',encoding='UTF-8')as f:
    for i in output_train:
        f.writelines(i)