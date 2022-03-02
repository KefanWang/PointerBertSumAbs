import pandas as pd
path='wikihowSep.csv'
data=pd.read_csv(path)
title_fromcsv=data['title']
result=[]
with open("all_val.txt",'r',encoding='UTF-8') as f:
    for line_val in f:
        result.append(list(line_test.strip('\n').split(",")))
output_val=[]


for i in result:
    for j in range(len(title_fromcsv)):
        output=[]
        if type(title_fromcsv[j])==str:
            if i==title_fromcsv[j].replace(" ",""):
                output.append(data.iloc[j]["overview"])
                output.append(data.iloc[j]["headline"])
                output.append(data.iloc[j]["text"])
                output.append(data.iloc[j]["sectionLabel"])
            
                output_val.append(output)

with open("outputval.txt",'w',encoding='UTF-8')as f:
    for i in output_val:
        f.writelines(i)
