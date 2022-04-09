#For Python 3.9.7

#Codechef solution crawler
#By Joss Moo-Young 100586602
#This app scrubs from a file to gather source code hosted on CodeChef.com

import io
from ast import Num
import requests # To connect to sites and get html from url
from bs4 import BeautifulSoup, SoupStrainer # To parse html, and to filter for only certain types of info in pages
import webbrowser # So I can open pages in browser to watch it work for fun (and debugging)
from datetime import datetime, timedelta # for logging
import urllib.parse # to allow us to parse relative paths into urls, and avoid trying to follow links other than http
from collections import deque # for discovered but unvisited sites
import os # for file i/o
#import validators # to check for malformed urls
import csv

##########################################################
##################### Helper Methods #####################

def normalizeCode(code: str):
    code = code.encode('ascii', 'ignore')
    return code

#For initial processing of urls. Percent-encode special characters and set all lower case
def normalizeURL(found_url_raw):
    found_url_normalized = found_url_raw.lower()#convert to lowercase to avoid duplicates like Uoit.ca and uoit.ca, also replace commas in urls since they are needed as delimiters in the csv output
    found_url_normalized = urllib.parse.quote(found_url_normalized, safe=(':/\\;?a=&$:@+'))
    return found_url_normalized

def contains(toSearch, criteria):
    for thing in criteria:
        if thing in toSearch:
            return True
    return False
 
def writeFile(path: str, data: str):
    file = io.open(path, 'w', encoding='utf-8')
    file.write(data)
    file.close()

def appendFile(path: str, data: str):
    file = io.open(path, 'a', encoding='utf-8')
    file.write(data)
    file.close()

#Because I needed a LOT of debugging...
def log(string):
    print(string + '\n')
    appendFile(logFilePath, string + '\n')

##################### Helper Methods #####################
##########################################################




####################################################
##################### Settings #####################
allowRedirects = True
#Just so I can see it working (mostly for fun)
openInBrowser = False 
##################### Settings #####################
####################################################



####################################################
##################### Crawler ######################


startTime = datetime.now() #for stats afterward
#Create output files
startTimeString = startTime.strftime('%d-%m-%Y_%H-%M-%S')
logFilePath = "D:/Projects/Code_Style_Similarity/logs/log_" + startTimeString + ".txt" #logging info
inputFilePath = 'D:/Projects/Code_Style_Similarity/archive/solutions.csv'
outputFolderPath = 'D:/Projects/Code_Style_Similarity/output'
outputTableFilePath = 'D:/Projects/Code_Style_Similarity/codeIndexTable.csv' #holds reference to each source file. Contains usernames

writeFile(logFilePath, startTimeString + '\n')


#Prompt user to enter a file to load to resume
if(inputFilePath == ''):
    print('Enter a file to load, or blank to start again')
    inputFilePath = input()

with open(outputTableFilePath, 'w+', newline='') as outputcsv:
    columnNames = ['QCode', 'UserID', 'SolutionID', 'SourceFile']
    writer = csv.DictWriter(outputcsv, fieldnames = columnNames)
    writer.writeheader()
    outputcsv.close()
    try:
        with open(inputFilePath, newline='') as inputcsv:
            reader = csv.DictReader(inputcsv)
            rows = list(reader)
            totalrows = len(rows)
            for rowNum in range(436, totalrows):
                row = rows[rowNum]
                qCode = row['QCode']
                solutionID = row['SolutionID']
                userID = os.path.basename(row['UserID'])
                status = row['Status']
                language = row['Language']
                url = urllib.parse.urljoin('https://www.codechef.com/', row['SolutionUrl'])

                if(language != 'JAVA' or status == 'compilation error'):
                    continue
                
                log('======================================================\n' + 'Visiting url: ' +  url)
                log('row no. ' + str(rowNum)) 
                #log('row no. ' + str(rowNum) + ' of ' + str(totalrows)) 
                #Main crawl
                try:
                    response = requests.get(url, timeout=5, allow_redirects=allowRedirects)
                    if openInBrowser: webbrowser.open(url, new=0)
                except Exception as e:
                    log('failed to connect:\n\turl: ' + str(url) + '\n\texception: ' + str(e))
                    continue

                #parse pages with BeautifulSoup only for links
                htmltext = response.text
                strainer = SoupStrainer('pre') #search for <pre> tag
                parsedPage = BeautifulSoup(htmltext, 'html.parser', parse_only=strainer)
                codeText = parsedPage.text

                outputFileBasename = (qCode + '_' + userID + '_' + solutionID + '.java')
                outputFilePath = os.path.join(outputFolderPath, outputFileBasename)
                writeFile(outputFilePath, codeText)

                outputcsv = open(outputTableFilePath, 'a', newline='')
                writer = csv.DictWriter(outputcsv, fieldnames = columnNames)
                writer.writerow({'QCode': qCode, 'UserID': userID, 'SolutionID': solutionID, 'SourceFile': outputFileBasename})
                outputcsv.close()       
                # log('Visited url: ' +  url +'\n======================================================') 
        endTime = datetime.now()
        duration = (endTime - startTime)
        print('Finished')
        log('\n\n\n===COMPLETE===\n\n\n')
        log('time to complete: ' + str(duration) + '\n')

    except KeyboardInterrupt as e:
        log('\nXXXXXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXXXX\n Aborted due to exception: ' + str(e))
        print('\nAborted: ' + outputFilePath + '\n')

##################### Crawler ######################
####################################################
