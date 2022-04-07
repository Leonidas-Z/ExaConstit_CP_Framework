import logging


class Logger:
    def __init__(self):

        # Make log file to track the runs. This file will be created after the code starts to run.
        level = logging.DEBUG
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