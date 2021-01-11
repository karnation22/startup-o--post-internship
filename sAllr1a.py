import pandas as pd  
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA 

NUMER_ORD = [3,4,5,6,26,27,31,43,44]
CAT_ORD = [7,8,14,28,29,45,47]
BIN_PRES = [9,10,11,12,16,17,18,19,20]
ONE_HOT = [22]
COMM = [13,21,24,37,42]
OTHER = [48,49,50,51]
sia = SIA()
sheet_names = pd.ExcelFile("BaseData.xlsx").sheet_names
sheet_name_main_L = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if(i%4==0)]
sheet_name_results_L = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if (i%4==1)]
weight_files_L = ["s2r1a-Weights.csv","s3r1a-Weights.csv","s4r1a-Weights.csv",
        "s5r1a-Weights.csv","s7r1a-Weights.csv","s8r1a-Weights.csv"]
output_files_L = ["s2r1a.csv","s3r1a.csv","s4r1a.csv","s5r1a.csv","s7r1a.csv","s8r1a.csv"]
REL_COLS = []
REL_COLS.extend(NUMER_ORD)
REL_COLS.extend(CAT_ORD)
REL_COLS.extend(BIN_PRES)
REL_COLS.extend(ONE_HOT)
REL_COLS.extend(COMM)
REL_COLS.extend(OTHER)
REL_COLS = sorted(REL_COLS)
NAME_COMPANY = [1,2]


def main_sALLa(NUMER_ORD,CAT_ORD,BIN_PRES,ONE_HOT,COMM,OTHER,REL_COLS):

    for(sheet_name_main,sheet_name_result,output_file, weight_file) in \
        zip(sheet_name_main_L,sheet_name_results_L,output_files_L, weight_files_L):
        data = pd.read_excel("BaseData.xlsx",  header=0, sheet_name=sheet_name_main)
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
                if("med" in x.lower()): return  0.5
                elif("high" in x.lower()): return 1.0
                else: return 0
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_CAT_ORD(x))
        for i in BIN_PRES:
            data.iloc[:,i] = data.iloc[:,i].apply(lambda x: int(pd.isna(x)))

        for i in COMM:
            def helper_COMM(x):
                if(pd.isna(x)): return 0
                else: return (1.0+float(sia.polarity_scores(x)['compound']))/2
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
        data4.to_csv(weight_file)
        bordaScoreCompanies = borda_score(sheet_name_result)
        name_diffL = {} #name --> [list of diff..]
        for index, row in data4.iterrows():
            name, company = row[0], row[1]
            row = row[2:].tolist()
            row = [float(item) for item in row]
            avg_row = sum(row)/len(row)
            if(name not in name_diffL): name_diffL[name] = []
            borda_company_score = bordaScoreCompanies[company]
            name_diffL[name].append(abs(avg_row-borda_company_score))
        for name in name_diffL:
            name_diffL[name] = sum(name_diffL[name])/len(name_diffL[name])
        newNameDictForPD = dict()
        newNameDictForPD['Name'] = []
        newNameDictForPD['Score'] = []
        for name in name_diffL:
            newNameDictForPD['Name'].append(name)
            newNameDictForPD['Score'].append(round(name_diffL[name],4))
        newNamePD = pd.DataFrame.from_dict(newNameDictForPD)
        newNamePD.to_csv(output_file)
    return 


def borda_score(sheet_name_result):
    data = pd.read_excel("BaseData.xlsx",sheet_name=sheet_name_result)
    header = data.iloc[0]
    data = data[1:]
    data.columns = header
    companyList = data['Company'].tolist()
    companyLen = len(companyList)
    retDict = {}
    ## 0 to 46
    for (i,company) in enumerate(companyList):
        companyScore = float(companyLen-1-i)/float(companyLen-1)
        retDict[company] = companyScore

    pdDict = dict()
    pdDict['Company'] = []
    pdDict['Score'] = []
    for company in retDict:
        score = round(retDict[company], 4)
        pdDict['Company'].append(company)
        pdDict['Score'].append(score)
    pd_frame = pd.DataFrame.from_dict(pdDict)

    stem = sheet_name_result[:sheet_name_result.index("(")].strip()
    pd_frame.to_csv(stem+"-Score.csv")
    return retDict

main_sALLa(NUMER_ORD,CAT_ORD,BIN_PRES,ONE_HOT,COMM,OTHER,REL_COLS)