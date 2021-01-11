import pandas as pd  
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA 


sia = SIA()

NUMER_ORD = [3,4,5,6,26,27,31,43,44]
CAT_ORD = [7,8,14,28,29,45,47]
BIN_PRES = [9,10,11,12,16,17,18,19,20]
ONE_HOT = [22]
COMM = [13,21,24,37,42]
OTHER = [48,49,50,51]
REL_COLS = []
REL_COLS.extend(NUMER_ORD)
REL_COLS.extend(CAT_ORD)
REL_COLS.extend(BIN_PRES)
REL_COLS.extend(COMM)
REL_COLS.extend(OTHER)
REL_COLS = sorted(REL_COLS)
NAME_COMPANY = [1,2]

sheet_names = pd.ExcelFile("BaseData.xlsx").sheet_names
sheet_names_main = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if i%4==0]

sheet_names_results = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if i%4==1]


def main():
    data4_list = []
    for sheet_name in sheet_names_main:
        data = pd.read_excel("BaseData.xlsx", sheet_name=sheet_name, header=0)
        new_header = data.iloc[0] 
        data = data[1:] 
        data.columns = new_header
        data2 = data.iloc[:,REL_COLS]
        data3 = data.iloc[:,NAME_COMPANY]
        for i in NUMER_ORD:
            MAX,MIN = data.iloc[:,i].min(), data.iloc[:,i].max()
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: (x - MIN)/(MAX - MIN) )
        for i in CAT_ORD:
            def helper_CAT_ORD(x):
                if("med" in x.lower()): return 0.5
                elif("high" in x.lower()): return 1.0
                else: return 0
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_CAT_ORD(x))
        for i in BIN_PRES:
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: int(pd.isna(x)))
        for i in COMM:
            def helper_COMM(x):
                if(pd.isna(x)): return 0.5
                else: return (float(sia.polarity_scores(x)['compound'])+1)/2
            try: data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_COMM(x))
            except: continue
        for i in OTHER:
            if(i==48):
                def helper_48(x):
                    if("no tech" in x.lower()): return 0
                    elif("differentiated" in x.lower()): return 0.5
                    else: return 1.0
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_48(x))
            elif(i==49):
                def helper_49(x):
                    if("conditional" in x.lower()): return 0.0
                    elif("logical" in x.lower()): return 0.5
                    else: return 1.0
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_49(x))
            elif(i==50):
                def helper_50(x):
                    if("early" in x.lower()): return 0.0
                    elif("imminent" in x.lower()): return 0.5
                    else: return 1.0
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_50(x))
            elif(i==51):
                def helper_51(x):
                    if("not differentiated" in x.lower()): return 0.0
                    elif("unique" in x.lower()): return 0.5
                    else: return 1.0
                data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_51(x))

        for (ind,j) in enumerate(REL_COLS):
            data2.iloc[:,ind] = data.iloc[:,j]
        data4 = pd.concat([data3, data2], axis=1)
        data4_list.append(data4)
    total = 0
    for data in data4_list:
        row,_ = data.shape
        total += row
    data_cols = data4_list[1].columns
    data_cols = data_cols[2:]
    expert_count = dict()
    expertIndexColScore = dict()
    for data in data4_list:
        for index, row in data.iterrows():
            name, company = row[0], row[1]
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
    allColumnKeys = list(expertScoreIndexCol.keys())
    for expert in expertIndexColScore:    
        curColumnKeys = list()
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
            curColumnKeys.append(column)
        for remainColumn in list(set(allColumnKeys)-set(curColumnKeys)):
            expertIndexColScore[expert][remainColumn] = 1.0
    pd_exp_ind_col_score = pd.DataFrame.from_dict(expertIndexColScore).transpose()
    
    if("s234578r1c.csv" in os.listdir()): os.remove("s234578r1c.csv")
    pd_exp_ind_col_score.to_csv("s234578r1c.csv")

main()