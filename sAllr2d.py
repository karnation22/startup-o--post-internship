from sklearn.linear_model import LinearRegression as LR 
import pandas as pd
import numpy as np
from strsimpy.levenshtein import Levenshtein

CSV_FILES = ["s2r2aWeights.csv","s3r2aWeights.csv","s4r2aWeights.csv",
        "s5r2aWeights.csv","s7r2aWeights.csv","s8r2aWeights.csv"]
SCORE_FILES = ["S2-R2-Score.csv","S3-R2-Score.csv","S4-R2-Score.csv","S5-R2-Score.csv",
            "S7-R2-Score.csv","S8-R2-Score.csv"]
WEIGHTS = "s234578r2c.csv"
LR = LR()
Levenshtein = Levenshtein()
def main():
    otherBool = False
    weightPandas = pd.read_csv(WEIGHTS)
    for csv_file,score_file in zip(CSV_FILES,SCORE_FILES):
        csvPandas = pd.read_csv(csv_file)
        scorePandas = pd.read_csv(score_file)
        columns = csvPandas.columns
        superPandasFrame = list()
        y_score_ans = list()
        for index, row in csvPandas.iterrows():
            if("S5" not in score_file): name,company = row[1], row[2]
            else: name, company = row[2], row[1]
            if(pd.isna(name) or pd.isna(company)): continue
            rowLeft = row[3:].tolist()
            isWeightColumn = weightPandas[weightPandas.columns[0]]==name
            relWeightColumn = weightPandas[isWeightColumn].to_dict()
            subPandasFrame = list()
            scoreOther = list()
            ColList = list()
            for index2, score in enumerate(rowLeft):
                relCol = columns[index2+3]
                if(relCol not in relWeightColumn): continue
                else: 
                    score *= list(relWeightColumn[relCol].items())[0][1]
                    subPandasFrame.append(score)
                    ColList.append(relCol)
            subPandasFrame = [np.mean(item) for item in subPandasFrame] ##iron out 'Other'
            superPandasFrame.append(subPandasFrame)
            scorePDScore = scorePandas['Score'].tolist()
            scorePDCompany = scorePandas['Company'].tolist()
            minDist,bestScore,flag = None, None, False
            for(score,iterCompany) in zip(scorePDScore, scorePDCompany):
                if(minDist==None or Levenshtein.distance(company.lower(),iterCompany.lower())<minDist):
                    if(minDist==0): # we have found an exact match.. 
                        y_score_ans.append(float(score))
                        flag = True
                        break
                    else: 
                        bestScore = score 
                        minDist = Levenshtein.distance(company.lower(),iterCompany.lower())
            if(not(flag)): y_score_ans.append(bestScore)
            
        newPD = pd.DataFrame(superPandasFrame,columns=ColList)
        newXData = newPD.to_numpy()
        yScoreAns = np.asarray(y_score_ans)
        reg = LR.fit(newXData,yScoreAns)
        coefficients = reg.coef_
        allVals = dict()
        allVals['Column'] = []
        allVals['RelWeight'] = []
        for coefficient, column in zip(coefficients, ColList):
            numerator = float(np.exp(coefficient))
            denominator = sum([float(np.exp(coeff)) for coeff in coefficients])
            allVals['Column'].append(column)
            allVals['RelWeight'].append(round(numerator/denominator, 4))
        allValsPD = pd.DataFrame.from_dict(allVals)
        strName = csv_file[:csv_file.index("a")]+"d.csv"
        allValsPD.to_csv(strName)
        
main()