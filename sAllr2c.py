import pandas as pd  
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA 


sia = SIA()

NAME_COL = [1,2]
NUMER_ORD = [3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19]
CAT_ORD = [14,27,29]
BIN_PRES = [20,21,22,23,24]
COMM = [25,26]
REL_COLS = []
REL_COLS.extend(NUMER_ORD)
REL_COLS.extend(CAT_ORD)
REL_COLS.extend(BIN_PRES)
REL_COLS.extend(COMM)
REL_COLS = sorted(REL_COLS)

sheet_names = pd.ExcelFile("BaseData.xlsx").sheet_names
sheet_names_main = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if i%4==2]
sheet_names_results = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if i%4==3]


def main():
    data4_list = []
    for sheet_name in sheet_names_main:
        data = pd.read_excel("BaseData.xlsx", sheet_name=sheet_name, header=0)
        new_header = data.iloc[0] 
        data = data[1:] 
        data.columns = new_header

        data2 = data.iloc[:,REL_COLS]
        data3 = data.iloc[:,NAME_COL]
        for i in NUMER_ORD:
            MAX,MIN = data.iloc[:,i].min(), data.iloc[:,i].max()
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: (x - MIN)/(MAX - MIN) )
        for i in CAT_ORD:
            if(i==14):
                def helper_14(x):
                    if("somewhat" in str(x).lower()): return 0.5
                    elif("highly" in str(x).lower()): return 1.0
                    else: return 0.0
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_14(x))
            if(i==27):
                def helper_27(x):
                    if("can be there" in str(x).lower()): return 1.0
                    elif("no chance" in str(x).lower()): return 0.0
                    else: return 0.5
                try: data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_27(x))
                except: continue
            if(i==29):
                def helper_29(x):
                    if("low" in str(x).lower()): return  0.0
                    elif("high" in str(x).lower()): return 1.0
                    else: return 0.5
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_29(x))
        for i in BIN_PRES:
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: int(pd.isna(x)))
        for i in COMM:
            def helper_COMM(x):
                if(pd.isna(x)): return 0.5
                else: return (float(sia.polarity_scores(x)['compound'])+1)/2
            try: data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_COMM(x))
            except: continue

        for (ind,j) in enumerate(REL_COLS):
            data2.iloc[:,ind] = data.iloc[:,j]
        data4 = pd.concat([data3, data2], axis=1)
        data4_list.append(data4)

    data_cols = data4_list[1].columns
    data_cols = data_cols[2:]
    expert_count = dict()
    expertIndexColScore = dict()
    for sheet_name,data in zip(sheet_names_main,data4_list):
        for index, row in data.iterrows():
            if("S5" in sheet_name): name, company = row[1], row[0]
            else: name, company = row[0], row[1]
            if(name not in expert_count): 
                expert_count[name] = 1
            else: expert_count[name] += 1
            rowNew = row[2:].tolist()
            for index,item in enumerate(rowNew):
                if(name not in expertIndexColScore): 
                    expertIndexColScore[name] = dict()
                else:
                    if(data_cols[index] not in expertIndexColScore[name]): 
                        expertIndexColScore[name][data_cols[index]] = [item]
                    else: expertIndexColScore[name][data_cols[index]].append(item)
    expertScoreIndexCol = dict()
    for expert in expertIndexColScore:
        for column in expertIndexColScore[expert]:
            if(column not in expertScoreIndexCol):
                expertScoreIndexCol[column] = []
            expertScoreIndexCol[column].extend(expertIndexColScore[expert][column])
    allColumnKeys = expertScoreIndexCol.keys()
    for expert in expertIndexColScore:
        currentKeys = list()
        for column in expertIndexColScore[expert]:

            relScoreIndexCol = expertScoreIndexCol[column]
            relScoreIndexCol = [item for item in relScoreIndexCol  if ~np.isnan(item)]
            numerator = np.mean(relScoreIndexCol)

            relIndexColScore = expertIndexColScore[expert][column]
            relIndexColScore = [item for item in relIndexColScore if ~np.isnan(item)]
            denominator = np.mean(relIndexColScore)

            if(abs(denominator-0.0)>=1e-5): 
                expertIndexColScore[expert][column] = round(float(numerator/denominator),4)
            else: expertIndexColScore[expert][column] = 1.0
            currentKeys.append(column)
        for remainColumn in (set(allColumnKeys)-set(currentKeys)):
            expertIndexColScore[expert][remainColumn] = 1.0

    newExpertIndexColScore = dict()
    for item in expertIndexColScore:
        if(pd.isna(item)): continue
        else: newExpertIndexColScore[item] = expertIndexColScore[item]

    pd_exp_ind_col_score = pd.DataFrame.from_dict(newExpertIndexColScore).transpose()

    if("s234578r2c.csv" in os.listdir()): os.remove("s234578r2c.csv")
    pd_exp_ind_col_score.to_csv("s234578r2c.csv")

main()