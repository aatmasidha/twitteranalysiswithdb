import logging
import logging.config as logconfig
from jproperties import Properties
import os
import configparam.constants as constants

# logconfig.fileConfig('./logging.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)
configs = Properties()

def readConfigurationFile():
    configParam = {}
    
    #to get the current working directory
    directory = os.getcwd()
    print(directory)
    jsonPath = ''
    with open('./configparam/app-config.properties', 'rb') as config_file:
        configs.load(config_file)
        # bearer_token = configs.get("MY_APP")
        pathKey = configs.get(constants.JASON_PATH)
        # If property is not defined in the .properties file
        if(pathKey != None): 
        
            # No value assigned to a property defined in .properties file
            jsonPath = configs[constants.JASON_PATH].data
            if(not jsonPath):
                jsonPath = "."
        configParam[constants.JASON_PATH] = jsonPath
        
        
        outputPathKey = configs.get(constants.OUTPUT_PATH)
        # If property is not defined in the .properties file
        if(outputPathKey != None): 
        
            # No value assigned to a property defined in .properties file
            outputPathKey = configs[constants.OUTPUT_PATH].data
            if(not outputPathKey):
                outputPathKey = "."
        configParam[constants.OUTPUT_PATH] = outputPathKey
        
        
        userNames = configs.get(constants.USER_NAMES)
        # If property is not defined in the .properties file
        if(userNames != None): 
            # No value assigned to a property defined in .properties file
            userNames = configs[constants.USER_NAMES].data
            if(not userNames):
                userNames = constants.TWITTER_USER_NAMES
        else:
            userNames = constants.TWITTER_USER_NAMES
        
        configParam[constants.USER_NAMES] = userNames
        
    return configParam

def main():
    readConfigurationFile()

if __name__ == "__main__":
    main()
