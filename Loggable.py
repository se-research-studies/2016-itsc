#!/usr/bin/python2

class Loggable:
    def __init__(self, logger, log_prefix):
        self.logger = logger
        self.log_prefix = log_prefix
    
    def debug(self,message):
        self.logger.debug('[' + self.log_prefix + '] ' + message)
    
    def info(self,message):
        self.logger.info('[' + self.log_prefix + '] ' + message)
