from datetime import datetime
import json
import logging
import os

# from bs4 import BeautifulSoup
from jproperties import Properties

from tweetdataextract import getnewslikes
from tweetdataextract import getnewsretweets
from tweetdataextract import getnewsbyhandler
from tweetdataextract import getnewscomments
from tweetdataextract import getuserinfo
import configparam.readconfigparam as readConfig
import configparam.constants as constants


logging.config.fileConfig('./configparam/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
configs = Properties()
  
def getNewsDataFromTwitterHandlers():
    
    jsonPath = "."
    logger.info('Started')
    configParam = readConfig.readConfigurationFile()
    jsonPath = configParam[constants.JASON_PATH]
   
    
    # Get information about the news handlers mentioned in the list 
    # by  userNames
    # Input Parameters is List of all the news handlers we have to analyse
    # output json formatted twitter account details for the news handlers
    # userNames = "the_hindu,AJEnglish,BBCWorld"
    userNames = configParam[constants.USER_NAMES]
    # userNames = "HelsinkiTimes"
    tweeterUserInformationJson = getuserinfo.getTweeterUserInformation(userNames)
    logger.debug(json.dumps(tweeterUserInformationJson, indent=4, sort_keys=True))

    for newsAccountRecord in tweeterUserInformationJson['data']:
        newsUserDict = {}
        newsUserAccountDict = {}
        newsDetailsDict = {}

        userName = newsAccountRecord[constants.TWITTER_USER_NAME]
        userId = newsAccountRecord['id']
        newsUserAccountDict["id"] = userId
        newsUserAccountDict['newsaccounthandler'] = userName
        executionDateTime = datetime.now()        
        newsUserAccountDict['capturedate'] = str(executionDateTime)
        newsUserAccountDict['location'] = newsAccountRecord['location']
        newsDetailsDict['newsAccountDetails'] = newsUserAccountDict
        newsUserDict[userId] = newsDetailsDict

        try:
              for userID, userData in newsUserDict.items():
                newsLikesDict = {}
                newsRetweetDict = {}
                newsCommentDict = {}

                userInfo = userData['newsAccountDetails']
                print (userInfo)     

                jsonString = json.dumps(userInfo)

                # Get the latest news for the user
                # Input name of the handler of the news Paper
                news_user_response = getnewsbyhandler.getNewsTweetInformationByUser(userInfo['newsaccounthandler'])
                for newsRecord in news_user_response['data']:
                    newsLikesDict = {}
                    newsRetweetDict = {}
                    newsCommentDict = {}
                    
                    
                    newsText = newsRecord['text'].encode("ascii", "ignore")
                    newsText = newsText.decode()        
                    newsId = newsRecord['id']
                    newsIDKey = 'newsId_' + str(newsId)
                    newsDetailsDict[newsIDKey] = {}

                    newsDetailsDict[newsIDKey]['id'] = newsId
                    newsDetailsDict[newsIDKey]['text'] = newsText
                    
                    # newsDetailsDict[newsIDKey]['location'] = newsRecord['location']

                    jsonString = json.dumps(newsDetailsDict)

                    logger.debug("newsID:" + newsId)
                    logger.debug("news Text:" + newsText)
                    # Retweet information for a news
                    retweet_responseJson = getnewsretweets.getNewsReTweetInformationByUser(newsId)
                    if retweet_responseJson['meta']['result_count'] != 0:
                        for retweet in retweet_responseJson['data']:
                            reTweetID = retweet['id']
                            reTweetUserName = retweet[constants.TWITTER_USER_NAME].encode("ascii", "ignore")
                            reTweetUserName = reTweetUserName.decode() 

                            reTweetName = retweet[constants.TWITTER_NAME].encode("ascii", "ignore")
                            reTweetName = reTweetName.decode()
                            
                            # TODO LOcation remove commas
                            reTweetLocation  = ''
                            # if 'location' in retweet.keys():
                            #     reTweetLocation = retweet['location'].encode("ascii", "ignore")
                            #     reTweetLocation = reTweetLocation.decode()

                            logger.debug("Retweet Information:" + reTweetID + reTweetUserName + reTweetName)
                            retweetIDKey = 'retweetID_' + str(reTweetID)
                            newsRetweetDict[retweetIDKey] = {"retweetID": reTweetID, "reUserName":reTweetUserName, "reTweetName":reTweetName , "reTweetLocation" : reTweetLocation } 

                    newsDetailsDict[newsIDKey]['newsRetweetDict'] = newsRetweetDict   

                    newsLikes_responseJson = getnewslikes.getLikesForNewsTweetInformation(newsId)
                    if newsLikes_responseJson['meta']['result_count'] != 0:
                        for news_likes in newsLikes_responseJson['data']:
                            likesID = news_likes['id']
                            likesUserName = news_likes[constants.TWITTER_USER_NAME].encode("ascii", "ignore")
                            likesUserName = likesUserName.decode() 

                            likesName = news_likes[constants.TWITTER_NAME].encode("ascii", "ignore")
                            likesName = likesName.decode()
                            likesIDKey = 'likesID_' + str(likesID)

                            # TO DO  remove comma from location
                            likesLocation  = ''
                            # if 'location' in news_likes.keys():
                            #     likesLocation = news_likes['location'].encode("ascii", "ignore")
                            #     likesLocation = likesLocation.decode()

                            
                            newsLikesDict[likesIDKey] = {"likesID": likesID, "likesUserName": likesUserName, "LikesName": likesName, "Location":likesLocation}    
                            logger.debug("Likes Information:" + likesID + ":" + likesUserName + ":" + likesName + "Location" + ":" + likesLocation)

                    newsDetailsDict[newsIDKey]['newsLikesDict'] = newsLikesDict

                    comments_responseJson = getnewscomments.getCommentsInNewsTweetInformationByUser(newsId)
                    if comments_responseJson['meta']['result_count'] != 0:
                        for comment in comments_responseJson['data']:
                            commentUserId = comment['author_id']
                            
                            commentUserResponse = getnewsbyhandler.getNewsTweetInformationByUser(commentUserId)
                            commentId = comment['id']
                            commentText = comment['text'].encode("ascii", "ignore")
                            commentText = commentText.decode()
                            userLocation = ''
                            if 'includes' in commentUserResponse.keys():
                                if 'users' in commentUserResponse['includes'].keys():
                                    # if 'location' in commentUserResponse['includes']['users'].keys():
                                    userInfo = commentUserResponse['includes']['users']
                                    
                                    if 'location' in userInfo[0].keys():
                                        userLocation = userInfo[0]['location']

                            commentIDKey = 'commentID_' + str(commentId)
                            newsCommentDict[commentIDKey] = {"commentUserID": commentUserId, "commentID": commentId, "commentText": commentText, "location" : userLocation } 
                            logger.debug("Comment Information:" + commentUserId + commentId + commentText)

                        newsDetailsDict[newsIDKey]['newsCommentDict'] = newsCommentDict    
                    jsonString = json.dumps(newsDetailsDict)
                    newsUserDict[userId] = newsDetailsDict        

                directory = os.getcwd()
                print(directory)
                timeStr = executionDateTime.strftime("%H_%M_%S")        
                dateStr = (executionDateTime).strftime("%Y_%m_%d")    
                with open(jsonPath + "\\" + userId + "_" + dateStr + "_T_" + timeStr + ".json", 'w') as json_file:
                     json.dump(newsUserDict, json_file, indent=4)    

        except BaseException as err:
                    logger.error(f"Unexpected {err=}, {type(err)=}")
                    raise TypeError(f"Unexpected {err=}, {type(err)=}")


logging.info('Finished')

