from unittest import result
import pandas as pd
path='wikihowSep.csv'
data=pd.read_csv(path)
title_fromcsv=data['title']
print(type(title_fromcsv[0]))
result=[]
#print(title_fromcsv[0].replace(" ",""))
with open("all_test.txt",'r',encoding='UTF-8') as f:
    for line_test in f:
        result.append(list(line_test.strip('\n').split(",")))
output_test=[]

for i in result:
    for j in range(len(title_fromcsv)):
        output=[]
        if type(title_fromcsv[j])==str:
            if i==title_fromcsv[j].replace(" ",""):
                output.append(data.iloc[j]["overview"])
                output.append(data.iloc[j]["headline"])
                output.append(data.iloc[j]["text"])
                output.append(data.iloc[j]["sectionLabel"])
                print("AA")
                output_test.append(output)

with open("outputtest.txt",'w',encoding='UTF-8')as f:
    for i in output_test:
        f.writelines(i)  


