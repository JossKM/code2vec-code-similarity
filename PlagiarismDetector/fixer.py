#For Python 3.9.7

#Codechef solution crawler
#By Joss Moo-Young 100586602
#This app scrubs from a file to gather source code hosted on CodeChef.com

import io
from msilib.schema import Error # for file i/o
import os # for file i/o
import csv
from re import I

##################### Helper Methods #####################
##########################################################

#Create output files
inputFilePath = 'D:/Projects/code2vec/CodeChefCrawler/archive/solutions.csv'
vectorFilePath = 'D:/Projects/code2vec/outputs/vectors_09-04-2022_21-58-47.csv'
outputTableFilePath = 'D:/Projects/code2vec/outputs/vectors_09-04-2022_21-58-47-fixed.csv'

#Prompt user to enter a file to load to resume
if(inputFilePath == ''):
    print('Enter a file to load, or blank to start again')
    inputFilePath = input()

with open(outputTableFilePath, 'w+', newline='') as outputcsv:
    columnNames = ['QCode', 'UserID', 'SolutionID', 'SourceFile', 'Status', 'TimeTaken', 'MemTaken', 'CodeName', 'PredictedName', 'Vector']
    writer = csv.DictWriter(outputcsv, fieldnames = columnNames)
    writer.writeheader()
    outputcsv.close()

    with open(inputFilePath, newline='') as inputcsv:
        with open(vectorFilePath, newline='') as vectorFile:

            reader = csv.DictReader(inputcsv)
            rows = list(reader)
            totalrows = len(rows)
            
            vecReader = csv.DictReader(vectorFile)
            vectorRows = list(vecReader)
            vecRows = len(vectorRows)

            for vecRowNum in range(0, vecRows):
                vectorRow = vectorRows[vecRowNum]
                solutionID = vectorRow['SolutionID']

                row : list
                isShared = False
                for inputNum in range(0, totalrows):
                    if rows[inputNum]['SolutionID'] == solutionID:
                        isShared = True
                        row = rows[inputNum]

                if not isShared: continue

                if(row['SolutionID'] != solutionID):
                    raise Exception('Hello there')

                qCode = vectorRow['QCode']
                userID = os.path.basename(vectorRow['UserID'])
                outputFileBaseName = vectorRow['SourceFile']

                outputcsv = open(outputTableFilePath, 'a', newline='')
                writer = csv.DictWriter(outputcsv, fieldnames = columnNames)
                writer.writerow({'QCode': qCode, 'UserID': userID, 'SolutionID': solutionID, 'Status': row['Status'], 'TimeTaken': row['TimeTaken'], 'MemTaken': row['MemTaken'], 'SourceFile': outputFileBaseName, 'CodeName': vectorRow['CodeName'], 'PredictedName':vectorRow['PredictedName'], 'Vector':vectorRow['Vector']})
                  
##################### Crawler ######################
####################################################
