import logging
from msilib.schema import Error


class Logger:
    
    def __init__(self, loglevel = "DEBUG"):

        if loglevel == "DEBUG":
            loglevel = logging.DEBUG
        elif loglevel == "INFO":
            level = logging.INFO
        elif loglevel == "WARNING":
            level = logging.WARNING        
        elif loglevel == "ERROR":
            level = logging.ERROR
        else:
            raise Exception("Wrong loglevel input argument in Logger")

        # Make log file to track the runs. This file will be created after the code starts to run.
        logging.basicConfig(filename='logbook3_ExaProb.log', level=level, format='%(message)s', datefmt='%m/%d/%Y %H:%M:%S ', filemode='w')
        self.logger = logging.getLogger()



    def write_ExaProb_log(self, text, type='debug', changeline=False):
       
        if changeline == True:
            self.logger.info('\n')
            print('\n')

        if type=='error':
            self.logger.error('ERROR: '+text)
            print('ERROR: '+text)
        elif type =='warning':
            self.logger.warning('WARNING: '+text)
            print('WARNING: '+text)
        elif type =='info':
            self.logger.info(text)
            print(text)
        elif type == 'debug':
            self.logger.debug(text)
        else:
            raise Exception("Wrong type input argument in Logger")