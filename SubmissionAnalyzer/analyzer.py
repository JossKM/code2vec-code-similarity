from asyncio import proactor_events
from collections import namedtuple
import csv
from telnetlib import NOP
import tensorflow as tf
from tensorflow import norm
from tensorflow import losses
import numpy as np

#columnNamesProblems  = ['QCode', 'Mean', 'SolutionID', 'Distance', 'Radius', 'NumSolutions', 'MeanSuccess', 'DistSuccess', 'RadiusSuccess', 'NumSuccess', 'MeanWrong', 'DistWrong', 'RadiusWrong', 'NumWrong', 'MeanError', 'DistError', 'RadiusError', 'NumError']
columnNamesProblems  = ['QCode', 'Mean', 'SolutionID', 'Distance', 'NumSolutions', 'MeanSuccess', 'SolutionIDSuccess', 'DistSuccess', 'NumSuccess', 'MeanWrong', 'SolutionIDWrong', 'DistWrong', 'NumWrong', 'MeanError', 'SolutionIDError', 'DistError', 'NumError']
columnNamesSolutions = ['QCode', 'UserID', 'SolutionID', 'SourceFile',  'Status', 'TimeTaken', 'MemTaken', 'CodeName', 'PredictedName', 'Vector', 'MostSimilarCode', 'PlagiarismScore']
vectorTablePath = 'D:/Projects/code2vec/outputs/vectors_09-04-2022_21-58-47-fixed.csv'
outputTablePath = 'SubmissionAnalyzer/output/'


class ProblemDataEntry:
    qCode: str
    allSolutions: list #of CodeDataEntry
    
    mean: np.ndarray
    distances: list
  #  radius: np.float64
    numSolutions: int
    
    meanSuccess: np.ndarray
    distSuccess: list
   # radiusSuccess: np.float64
    numSuccess: int

    meanWrong: np.ndarray
    distWrong: list
 #   radiusWrong: np.float64
    numWrong: int

    meanError: np.ndarray
    distError: list
#    radiusError: np.float64
    numError: int

    def __init__(self, qCode: str):
        self.qCode = qCode
        self.allSolutions = list()

        self.distances = list()
        self.distSuccess = list()
        self.distWrong = list()
        self.distError = list()

        # self.radius = 0
        # self.radiusSuccess = 0
        # self.radiusWrong = 0
        # self.radiusError = 0

        self.numSolutions = 0
        self.numSuccess = 0
        self.numWrong = 0
        self.numError = 0


class CodeDataEntry:
    qCode: str
    userID: str
    solutionID: int
    sourceFile: str
    status: str
    timeTaken: str
    memTaken: str
    vector: np.ndarray
    similarities: list

    def __init__(self, qCode: str, userID: str, solutionID: int, sourceFile: str,  status: str,  timeTaken: str,  memTaken: str, vector: np.ndarray, similarities: list):
        self.qCode = qCode
        self.userID = userID
        self.solutionID = solutionID
        self.sourceFile = sourceFile
        self.status = status
        self.timeTaken = timeTaken
        self.memTaken = memTaken
        self.vector = vector
        self.similarities = similarities if similarities is not None else []

NUM_KEEP_SIMILAR = 4

allProblems = dict()#(str, ProblemDataEntry) #dict(str, list(CodeDataEntry))


def distanceBetween(vectorA: np.ndarray, vectorB: np.ndarray):
    displacement_vector = np.subtract(vectorA, vectorB)
    distance = np.sqrt(displacement_vector.dot(displacement_vector))#tf.norm(displacement_vector).numpy()
    return distance

def computeSimilarity(entryA: CodeDataEntry, entryB: CodeDataEntry):
    #cosine_sim = tf.losses.cosine_similarity(entryA.vector, entryB.vector)
    #return cosine_sim.numpy()
    return distanceBetween(entryA.vector, entryB.vector)

print('opening: ' + vectorTablePath + '\n')

# with open(outputTablePath, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames = columnNamesSolutions)
#     writer.writeheader()

problemFileName = outputTablePath + 'All_Analysis' + '.csv'
with open(problemFileName, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = columnNamesProblems)
    writer.writeheader()

#Load solutions into ProblemDataEntry
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
        codeData = CodeDataEntry(qCode=row['QCode'], userID=row['UserID'], solutionID=solutionIDInt, sourceFile=row['SourceFile'], status=row['Status'], timeTaken=row['TimeTaken'], memTaken=row['MemTaken'], vector=codeVector, similarities=list())
        if(not codeData.qCode in allProblems):
            allProblems[codeData.qCode] = ProblemDataEntry(qCode = codeData.qCode) #make a new list for this problem
        
        allProblems[codeData.qCode].allSolutions.append(codeData) #add to list
        #allCodeVectors.append(codeData)


#Go through every problem
problem: ProblemDataEntry
for problem in allProblems.values():
    print('working on question ' + problem.qCode + '...\n')
    solutionList = problem.allSolutions
    numSolutions = len(solutionList)
    problem.numSolutions = numSolutions

    #Compare every solution to this problem to every other one
    solutionA: CodeDataEntry
    solutionB: CodeDataEntry

    problem.mean        = np.full_like(solutionList[0], 0.0)
    problem.meanSuccess = np.full_like(solutionList[0], 0.0)
    problem.meanWrong   = np.full_like(solutionList[0], 0.0)
    problem.meanError   = np.full_like(solutionList[0], 0.0)

    for idxA in range(0, numSolutions):
        solutionA = solutionList[idxA]
        print('\tsolution ' + str(solutionA.solutionID))
        problem.mean = np.add(problem.mean, solutionA.vector)
        #write stats
        if solutionA.status.startswith('accept'):
            problem.numSuccess += 1
            problem.meanSuccess = np.add(problem.meanSuccess, solutionA.vector)
        elif solutionA.status.startswith('wrong'):
            problem.numWrong += 1
            problem.meanWrong = np.add(problem.meanWrong, solutionA.vector)
        elif solutionA.status.startswith('runtime') or solutionA.status.startswith('time limit'):
            problem.numError += 1
            problem.meanError = np.add(problem.meanError, solutionA.vector)
        else:
            print('unknown status \"' + solutionA.status + '\" for solutionID ' + str(solutionA.solutionID) + '\n')
            #raise Exception(('unknown status ' + solutionA.status))

        # #Check similarity with every other program in this set
        # for idxB in range(idxA + 1, numSolutions):
        #     solutionB = solutionList[idxB]

        #     similarityScore = computeSimilarity(solutionA, solutionB)
        #     #Assign similarity scores
        #     solutionA.similarities.append((similarityScore, solutionB.solutionID))
        #     solutionB.similarities.append((similarityScore, solutionA.solutionID))

    #Check how similar each solution is to the mean
    # solution: CodeDataEntry
    # for solution in solutionList:
    #     solution.similarities.sort(reverse=False, key=lambda i: i[0])
    #     #solution.similarities = solution.similarities[0 : NUM_KEEP_SIMILAR]

    #     #write the similarity information for each problem

    #     outputFile = outputTablePath + str(problem.qCode) + '_Solutions.csv'
    #     with open(outputFile, 'a', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames = columnNamesSolutions)
            
    #         newRow = dict()
    #         newRow['QCode'] = solution.qCode
    #         newRow['UserID'] = solution.userID
    #         newRow['SourceFile'] = solution.sourceFile
    #         newRow['SolutionID'] = solution.solutionID
    #         newRow['Vector'] = ' '.join(map(str, solution.vector))
    #         newRow['MostSimilarCode'] = ' '.join(map(str, solution.similarities))
    #         newRow['PlagiarismScore'] = solution.similarities[0][0]
    #         writer.writerow(newRow)
    #         csvfile.close()

    #Calculate true means by dividing out sums
    

    problem.mean        = problem.mean          / problem.numSolutions
    if(problem.numSuccess > 0):
        problem.meanSuccess = problem.meanSuccess   / problem.numSuccess
    if(problem.numWrong > 0):
        problem.meanWrong   = problem.meanWrong     / problem.numWrong
    if(problem.numError > 0):
        problem.meanError   = problem.meanError     / problem.numError

    #Now we can compute distances from the mean...
    # problem.distSuccess = distanceBetween(problem.meanSuccess, problem.mean)
    # problem.distWrong   = distanceBetween(problem.meanSuccess, problem.mean)
    # problem.distError   = distanceBetween(problem.meanSuccess, problem.mean)

    for solution in problem.allSolutions:
        distance = distanceBetween(problem.mean, solution.vector)
        entry = (distance, solution.solutionID)
        problem.distances.append(entry)
        
        if solution.status.startswith('accept'):
            problem.distSuccess.append(entry)
        elif solution.status.startswith('wrong'):
            problem.distWrong.append(entry)
        elif solution.status.startswith('runtime') or solutionA.status.startswith('time limit'):
            problem.distError.append(entry)
       # else:
       #     print('unknown status \"' + solutionA.status + '\" for solutionID ' + str(solutionA.solutionID) + '\n')

    problem.distances.sort(reverse=False, key=lambda i: i[0]) #sort by most similar (smallest number)
    problem.distSuccess.sort(reverse=False, key=lambda i: i[0]) #sort by most similar (smallest number)
    problem.distWrong.sort(reverse=False, key=lambda i: i[0]) #sort by most similar (smallest number)
    problem.distError.sort(reverse=False, key=lambda i: i[0]) #sort by most similar (smallest number)


    #Write to file
    #problemFileName = outputTablePath + str(problem.qCode) + '_Analysis' + '.csv'
    problemFileName = outputTablePath + 'All_Analysis' + '.csv'
    with open(problemFileName, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = columnNamesProblems)
        writer.writeheader()

        newRow = dict()
        newRow['QCode'] = problem.qCode
        newRow['Mean'] = ' '.join(map(str, problem.mean))
        newRow['SolutionID'] = '[Problem Overall]'
        newRow['Distance'] = 0
        #newRow['Radius'] = problem.distances[len(problem.distances)-1]
        newRow['MeanSuccess'] = 'N/A' if problem.numSuccess < 1 else ' '.join(map(str, problem.meanSuccess))
        #newRow['DistSuccess'] = problem.distSuccess
        #newRow['RadiusSuccess'] = problem.distSuccess[len(problem.distSuccess)-1]
        newRow['NumSuccess'] = problem.numSuccess
        newRow['MeanWrong'] = 'N/A' if problem.numWrong < 1 else' '.join(map(str, problem.meanWrong))
        #newRow['DistWrong'] = problem.distWrong
        #newRow['RadiusWrong'] = problem.radiusWrong
        newRow['NumWrong'] = problem.numWrong
        newRow['MeanError'] = 'N/A' if problem.numError < 1 else' '.join(map(str, problem.meanError))
        #newRow['DistError'] = problem.distError
        #newRow['RadiusError'] = problem.radiusError
        newRow['NumError'] = problem.numError
        writer.writerow(newRow)
        csvfile.close()

    with open(problemFileName, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = columnNamesProblems)
        for distance in problem.distances:
            newRow = dict()
            newRow['SolutionID']        = distance[1]
            newRow['Distance']          = distance[0]
            writer.writerow(newRow)

        for distance in problem.distSuccess:
            newRow = dict()
            newRow['SolutionIDSuccess'] = distance[1]
            newRow['DistSuccess']       = distance[0]
            writer.writerow(newRow)

        for distance in problem.distWrong:
            newRow = dict()
            newRow['SolutionIDWrong']   = distance[1]
            newRow['DistWrong']         = distance[0]
            writer.writerow(newRow)

        for distance in problem.distError:
            newRow = dict()
            newRow['SolutionIDError']   = distance[1]
            newRow['DistError']         = distance[0]
            writer.writerow(newRow)

    csvfile.close()

print('done all jobs!\n')

#columnNamesProblems  = ['QCode', 'Mean', 'SolutionID', 'Distance', 'NumSolutions', 'MeanSuccess', 'SolutionIDSuccess', 'DistSuccess', 'NumSuccess', 'MeanWrong', 'SolutionIDWrong' 'DistWrong', 'NumWrong', 'MeanError', 'SolutionIDError' 'DistError', 'NumError']