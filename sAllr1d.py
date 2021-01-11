from sklearn.linear_model import LinearRegression as LR 
import pandas as pd
import numpy as np

CSV_FILES = ["s2r1aWeights.csv","s3r1aWeights.csv","s4r1aWeights.csv",
        "s5r1aWeights.csv","s7r1aWeights.csv","s8r1aWeights.csv"]
SCORE_FILES = ["S2-R1-Score.csv","S3-R1-Score.csv","S4-R1-Score.csv","S5-R1-Score.csv",
            "S7-R1-Score.csv","S8-R1-Score.csv"]
WEIGHTS = "s234578r1c.csv"
LR = LR()

def main():
    weightPandas = pd.read_csv(WEIGHTS)
    for csv_file,score_file in zip(CSV_FILES,SCORE_FILES):
        csvPandas = pd.read_csv(csv_file)
        scorePandas = pd.read_csv(score_file)
        columns = csvPandas.columns
        superPandasFrame = list()
        y_score_ans = list()
        for index, row in csvPandas.iterrows():
            name,company = row[1], row[2]
            rowLeft = row[3:].tolist()
            isWeightColumn = weightPandas[weightPandas.columns[0]]==name
            relWeightColumn = weightPandas[isWeightColumn].to_dict()
            subPandasFrame = list()
            scoreOther = list()
            ColList = list()
            for index2, score in enumerate(rowLeft):
                relCol = columns[index2+3]
                if("Other" in relCol): 
                    score *= list(relWeightColumn["Other"].items())[0][1]
                    scoreOther.append(score)
                elif(relCol not in relWeightColumn): continue
                else: 
                    score *= list(relWeightColumn[relCol].items())[0][1]
                    subPandasFrame.append(score)
                    ColList.append(relCol)
            subPandasFrame.append(scoreOther)
            ColList.append("Other")
            subPandasFrame = [np.mean(item) for item in subPandasFrame] ##iron out 'Other'
            superPandasFrame.append(subPandasFrame)
            isCompanyColumn = scorePandas['Company']==company
            relCompanyColumn = scorePandas[isCompanyColumn]
            y_score_ans.append(float(relCompanyColumn['Score']))
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
        ## print(allValsPD)
        strName = csv_file[:csv_file.index("a")]+"d.csv"
        allValsPD.to_csv(strName)
        
main()