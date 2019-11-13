"""
1.Download most recent version of slack client older version reference different module name
2.Create bot using slack apps and copy BOT Auth
3.Copy bot auth into Environment Variables
    Linux : export SLACK_BOT_TOKEN = [BOT'S TOKEN]

Useful Links:
https://python-slackclient.readthedocs.io/en/latest/basic_usage.html#joining-a-channel
https://python-slackclient.readthedocs.io/en/latest/real_time_messaging.html#connecting-to-the-rtm-api
https://github.com/SlackAPI/python-slackclient
https://api.slack.com/methods
https://api.slack.com/rtm
https://github.com/slackapi/python-slackclient/tree/master/tutorial
"""
import time
import os
from random import randrange
import slack

slack_token = os.environ['SLACK_BOT_TOKEN']
client = slack.WebClient(token=slack_token)
channel = "random"

#rtm_client = slack.RTMClient(token=slack_token)
#rtm_client.start()

#CHANNEL_LIST = client.api_call("channels.list").get("channels", [])
#ANSWERS_LIST = [
#    "Umm, interesting I will ask my master for its meaning",
#    "I think it cool",
#    "Howdy!",
#    "What's up?",
#    "I see",
#    "Anythings else?",
#    "I'm done!!!"
#]

#Example
client.chat_postMessage(
  channel=channel,
  text="Hello from your app! :tada:"
)

client.chat_postMessage(
  channel=channel,
  blocks=[
    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "Current weather in Auckland NZ is rain"
        }
    },
    {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "<https://openweathermap.org/|Open Weather Map> \n Rain	5.4 mm \n Wind:Moderate breeze, 6.7 m/s, North-northeast ( 30 )"
        },
        "accessory": {
            "type": "image",
            "image_url": "https://patch.com/img/cdn20/users/22992871/20190313/094116/styles/raw/public/processed_images/donatphotography_rain_umbrella_shutterstock_728383990_copy-1552484394-5009.jpg",
            "alt_text": "Rain"
        }
    },
    {
        "type": "section",
        "fields": [
            {
                "type": "mrkdwn",
                "text": "*Temperature*\n18 C"
            }
        ]
    }
  ]
)

def find_channels_id(channel_name, channel_list):
    """
        Find channel id by its name
        channel_name (string): name of finding channel
        channel_list (list): all channel list in the workspace
        return -> channel_id
    """
    match = [ch.get("id") for ch in channel_list if ch.get("name") == channel_name]
    return match[0]


if __name__ == "__main__":
    """
        Running Bot in this section
    """
    if rtm_client.connect(with_team_state=False):
        print("I'm online, please order me...")
        response_channels = find_channels_id("random", CHANNEL_LIST)
        # Example
        rtm_client.chat_postMessage(
            channel=response_channels,
            text="Hello sexy"
        )
        while True:
            data = client.rtm_read()
            if not data:
                continue
            else:
                new_data = data[0]
                print(new_data)
                if new_data.get("type") == "message" \
                    and new_data.get("subtype") != "bot_message":

                    client.api_call(
                        "chat.postMessage",
                        channel=response_channels,
                        text=ANSWERS_LIST[randrange(len(ANSWERS_LIST))]
                    )

            time.sleep(1)

    else:
        print("Something wrong, please check your internet connection!")
