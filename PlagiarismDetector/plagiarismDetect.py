from collections import namedtuple
import csv
import tensorflow as tf
from tensorflow import norm
from tensorflow import losses
import numpy as np

columnNames = ['QCode', 'UserID', 'SolutionID', 'SourceFile', 'CodeName', 'PredictedName', 'Vector', 'MostSimilarCode', 'PlagiarismScore']
vectorTablePath = '../outputs/vectors_08-04-2022_17-16-04.csv'
outputTablePath = 'similarity.csv'

class CodeDataEntry:
    qCode: str
    userID: str
    solutionID: int
    sourceFile: str
    vector: np.ndarray
    similarities: list

    def __init__(self, qCode: str, userID: str, solutionID: int, sourceFile: str, vector: np.ndarray, similarities: list):
        self.qCode = qCode
        self.userID = userID
        self.solutionID = solutionID
        self.sourceFile = sourceFile
        self.vector = vector
        self.similarities = similarities if similarities is not None else []


#For each vector,
#Find closest vectors
#similarityScores
NUM_KEEP_SIMILAR = 4

#solutionID, code vector
allCodeVectors = list()

def computeSimilarity(entryA: CodeDataEntry, entryB: CodeDataEntry):
    cosine_sim = tf.losses.cosine_similarity(entryA.vector, entryB.vector)
    return cosine_sim.numpy()

print('opening: ' + vectorTablePath + '\n')

with open(outputTablePath, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = columnNames)
    writer.writeheader()

with open(vectorTablePath, 'r', newline='') as csvRead:
    reader = csv.DictReader(csvRead)
    rows = list(reader)
    csvRead.close()
    totalrows = len(rows)

    #For each code file...
    for rowNum in range(0, totalrows):
        row = rows[rowNum]
        #print(str(row))
        solutionIDInt = int(row['SolutionID'].lstrip('S'))
        codeVectorString = row['Vector']
        codeVector = np.fromstring(codeVectorString, sep=' ')
        codeData = CodeDataEntry(qCode=row['QCode'], userID=row['UserID'], solutionID=solutionIDInt, sourceFile=row['SourceFile'], vector=codeVector, similarities=list())
        allCodeVectors.append(codeData)

        #print(str(codeVector) + '\n')
        # #compare with other files
        # displacement_vector = np.subtract(newCodeVector,lastCodeVector)
        # distance = tf.norm(displacement_vector)
        # #tf.print(distance)
        # print('Cosine Similarity with last code:\n')
        # cosine_sim = tf.losses.cosine_similarity(newCodeVector, lastCodeVector)
        # #tf.print(cosine_sim)

        #write the similarity information
        # with open(outputTablePath, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames = columnNames)
        #     row['Vector'] = ' '.join(map(str, codeVector))
        #     row['Similarity'] = ''
        #     row['MostSimilarCode'] = ''
        #     writer.writerow(row)
        #     csvfile.close()

numSolutions = len(allCodeVectors)
for idxA in range(0, numSolutions):
    for idxB in range(idxA + 1, numSolutions):
        solutionA = allCodeVectors[idxA]
        solutionB = allCodeVectors[idxB]

        if(solutionA.qCode == solutionB.qCode):
            if(solutionA.userID == solutionB.userID):
                continue #do not bother checking for plagiarism between a user and themselves on multiple attempts at the same problem

        similarityScore = computeSimilarity(solutionA, solutionB)
        #Assign similarity scores
        solutionA.similarities.append((similarityScore, solutionB.solutionID))
        solutionB.similarities.append((similarityScore, solutionA.solutionID))

solution: CodeDataEntry
for solution in allCodeVectors:
    solution.similarities.sort(reverse=False, key=lambda i: i[0])
    solution.similarities = solution.similarities[0 : NUM_KEEP_SIMILAR]

    #write the similarity information
    with open(outputTablePath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = columnNames)
        
        newRow = dict()
        newRow['QCode'] = solution.qCode
        newRow['UserID'] = solution.userID
        newRow['SourceFile'] = solution.sourceFile
        newRow['SolutionID'] = solution.solutionID
        newRow['Vector'] = ' '.join(map(str, solution.vector))
        newRow['MostSimilarCode'] = ' '.join(map(str, solution.similarities))
        newRow['PlagiarismScore'] = solution.similarities[0][0]
        writer.writerow(newRow)
        csvfile.close()

print('done!\n')