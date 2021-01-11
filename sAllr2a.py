import pandas as pd  
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA 


sheet_names = pd.ExcelFile("BaseData.xlsx").sheet_names
sheet_name_main_L = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if(i%4==2)]
sheet_name_results_L = [sheet_name for (i,sheet_name) in enumerate(sheet_names) if (i%4==3)]
weight_files_L = ["s2r2a-Weights.csv","s3r2a-Weights.csv","s4r2a-Weights.csv",
        "s5r2a-Weights.csv","s7r2a-Weights.csv","s8r2a-Weights.csv"]
output_files_L = ["s2r2a.csv","s3r2a.csv","s4r2a.csv","s5r2a.csv","s7r2a.csv","s8r2a.csv"]
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

def main():
    for(sheet_name_main,sheet_name_result,output_file, weight_file) in \
        zip(sheet_name_main_L,sheet_name_results_L,output_files_L, weight_files_L):
        data = pd.read_excel("BaseData.xlsx",sheet_name=sheet_name_main)
        new_header = data.iloc[0] 
        data = data[1:] 
        data.columns = new_header
        data2 = data.iloc[:, REL_COLS]
        data3 = data.iloc[:, NAME_COL]
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
                    if(pd.isna(x)): return 0
                    else: return (1.0+float(sia.polarity_scores(x)['compound']))/2

            try: data.iloc[:,i] = data.iloc[:,i].apply(lambda x: helper_COMM(x))
            except: continue
        for (ind,j) in enumerate(REL_COLS):
            data2.iloc[:,ind] = data.iloc[:,j]
        data4 = pd.concat([data3, data2], axis=1)
        data4.to_csv(weight_file)
        bordaScoreCompanies = borda_score(sheet_name_result)
        name_diffL = {} 
        for index, row in data4.iterrows():
            if("S5" in sheet_name_main): name, company = row[1], row[0]
            else: name, company = row[0], row[1]
            if(pd.isna(name) or pd.isna(company)): continue
            row = row[2:].tolist()
            row = [float(item) for item in row]
            avg_row = sum(row)/len(row)
            if(name not in name_diffL): name_diffL[name] = []

            try:
                borda_company_score = bordaScoreCompanies[company]
                name_diffL[name].append(abs(avg_row-borda_company_score))
            except:
                for bordaCompany in bordaScoreCompanies:
                    if(bordaCompany.lower() in str(company).lower() or 
                        str(company).lower() in bordaCompany.lower()):
                        borda_company_score = bordaScoreCompanies[bordaCompany]
                        name_diffL[name].append(abs(avg_row-borda_company_score))
                        break

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
    if("S5" not in sheet_name_result):
        header = data.iloc[0]
        data = data[1:]
        data.columns = header
    if("S3" in sheet_name_result or "S2" in sheet_name_result):
        header = data.iloc[0]
        data = data[1:]
        data.columns = header

    allCols = data.columns.tolist()
    relCols = list(filter(lambda x: "startup" in str(x).lower(), allCols ))

    companyList = data[relCols[0]].tolist()
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

main()