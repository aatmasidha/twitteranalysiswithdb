import json
import logging
import logging.config
import configparam.constants as constants
import os

import requests

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")


def getReTweetURLInformation(newsID):
   # return "https://api.twitter.com/2/users/by?usernames={}&user.fields=created_at&expansions=pinned_tweet_id".format(userNames)
   return "https://api.twitter.com/2/tweets/{}/retweeted_by".format(newsID)


def get_params():
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    return {"tweet.fields": "created_at", "user.fields" : "created_at,location"}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def getNewsReTweetInformationByUser(newsID):
    url = getReTweetURLInformation(newsID)
    params = get_params()
    # json_response = connect_to_endpoint(url, params)
    json_response = connect_to_endpoint(url, params)
    return json_response

    
def main():
    userID = "1528772793998970880"
    json_response = getNewsReTweetInformationByUser(userID)
    # print(json.dumps(json_response, indent=4, sort_keys=True))
    for record in json_response['data']:
         print(record['id'])
         print(record[constants.TWITTER_USER_NAME].encode('utf8'))
         print(record[constants.TWITTER_NAME].encode('utf8'))


if __name__ == "__main__":
    main()

