import traceback

from common import common
from extractor import Extractor
import numpy as np
#from typing import Optional
import tensorflow as tf
from tensorflow import norm
from tensorflow import losses
from datetime import datetime, timedelta # for logging
import os

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'

class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']
    delimiter = ','

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def writeFile(self, filepath: str, dataToWrite: str):
        file = open(filepath, 'w')
        file.write(dataToWrite)
        file.close()

    def appendFile(self, filepath: str, dataToWrite: str):
        file = open(filepath, 'a')
        file.write(dataToWrite)
        file.close()

    def log(self, string: str):
        self.appendFile(self.logFilePath, string + '\n')

    def predict(self):
        startTime = datetime.now() #for stats afterward
        #Create output files
        startTimeString = startTime.strftime('%d-%m-%Y_%H-%M-%S')
        logFilePath = "outputs/logs/log_" + startTimeString + ".txt" #logging info
        inputFilePath = os.path.join(os.getcwd(),'inputs')
        outputFilePath = 'outputs/vectors_' + startTimeString + '.csv' #results of crawl
        self.writeFile(filepath=logFilePath, dataToWrite=startTimeString + '\n')
        self.writeFile(filepath=outputFilePath, dataToWrite='')
        self.appendFile(filepath=outputFilePath, dataToWrite=self.delimiter.join(["codeIndex", "filename", "codeName", "predicted name", "vector"])+'\n')
        
        lastCodeVector = np.ndarray(0)
        print('Starting interactive prediction...')

        codeIndex = 0
        for input_filename in os.listdir(inputFilePath):        
            # print(
            #     'Modify the file: "%s" and press any key when ready, or "q" / "quit" / "exit" to exit' % input_filename)
            # user_input = input()
            # if user_input.lower() in self.exit_keywords:
            #     print('Exiting...')
            #     return
            try:
                predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(os.path.join(inputFilePath,input_filename))
            except ValueError as e:
                print(e)
                continue
            raw_prediction_results = self.model.predict(predict_lines)
            method_prediction_results = common.parse_prediction_results(
                raw_prediction_results, hash_to_string_dict,
                self.model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)
            for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
                print('Original name:\t' + method_prediction.original_name)
                for name_prob_pair in method_prediction.predictions:
                    print('\t(%f) predicted: %s' % (name_prob_pair['probability'], name_prob_pair['name']))
                print('Attention:')
                for attention_obj in method_prediction.attention_paths:
                    print('%f\tcontext: %s,%s,%s' % (
                    attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))
               
               
                if self.config.EXPORT_CODE_VECTORS:
                    vector_string = ' '.join(map(str, raw_prediction.code_vector))
                    
                    #write line
                    lineToWrite = self.delimiter.join([str(codeIndex), input_filename, method_prediction.original_name, str(method_prediction.predictions[0]['name']), vector_string])
                    self.appendFile(filepath=outputFilePath, dataToWrite=lineToWrite+'\n')
                   
                    newCodeVector = raw_prediction.code_vector
                    if(lastCodeVector.size > 0):
                        #compare vectors
                        print('Distance from last code:\n')
                        displacement_vector = np.subtract(newCodeVector,lastCodeVector)
                        distance = tf.norm(displacement_vector)
                        tf.print(distance)

                        print('Cosine Similarity with last code:\n')
                        cosine_sim = tf.losses.cosine_similarity(newCodeVector, lastCodeVector)
                        tf.print(cosine_sim)
                        
                    lastCodeVector = newCodeVector
            codeIndex += 1
