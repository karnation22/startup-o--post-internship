import pandas as pd

R1_SCORES = ["S2-R1-Score.csv","S3-R1-Score.csv","S4-R1-Score.csv",
    "S5-R1-Score.csv","S7-R1-Score.csv","S8-R1-Score.csv"]
R2_SCORES = ["S2-R2-Score.csv","S3-R2-Score.csv","S4-R2-Score.csv",
    "S5-R2-Score.csv","S7-R2-Score.csv","S8-R2-Score.csv"]
output_files = ["s2r2b.csv",'s3r2b.csv',"s4r2b.csv",
    's5r2b.csv',"s7r2b.csv","s8r2b.csv"]

def main():
    ## diffScore = dict()
    for(r1_score,r2_score, output_file) in zip(R1_SCORES,R2_SCORES,output_files):
        diffScore = dict()
        diffScore['Company'] = []
        diffScore['DiffScore'] = []
        curR1PD = pd.read_csv(r1_score)
        curR2PD = pd.read_csv(r2_score)
        curR1PDC = curR1PD['Company'].tolist()
        curR1PDS = curR1PD['Score'].tolist()
        curR1PDD = dict(zip(curR1PDC,curR1PDS))
        curR2PDC = curR2PD['Company'].tolist()
        curR2PDS = curR2PD['Score'].tolist()
        curR2PDD = dict(zip(curR2PDC,curR2PDS))
        for company in curR1PDD:
            if(company in curR2PDD):
                ## print("company: ",company)
                ## print(curR1PDD[company])
                ## print(curR2PDD[company])
                diffScore['Company'].append(company)
                diffScore['DiffScore'].append(float(curR1PDD[company]+curR2PDD[company])/2)
        pdDiffScore = pd.DataFrame.from_dict(diffScore)
        pdDiffScore.to_csv(output_file)

main()