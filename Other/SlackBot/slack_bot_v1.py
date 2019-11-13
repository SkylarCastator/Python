import os
import slack

@slack.RTMClient.run_on(event='message')
def say_hello(**payload):
    data = payload['data']
    web_client = payload['web_client']
    rtm_client = payload['rtm_client']
    if 'Test' in data.get('text', []):
        channel_id = data['channel']
        thread_ts = data['ts']
        user = data['user']

        web_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            icon_emoji=':robot_face:',
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Hi <@{user}>," + " I'm slack bot!"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "<https://python-slackclient.readthedocs.io/en/latest//|Slack API> \n This is all referenced in a block \n Best of luck!"
                    },
                    "accessory": {
                        "type": "image",
                        "image_url": "https://media.wired.com/photos/5933668e7965e75f5f3c7bca/191:100/pass/biglebowskytumb.jpg",
                        "alt_text": "Luck"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": "*Channel :  *\n" + channel_id
                        }
                    ]
                },
                {
                    "type": "actions",
                    "block_id": "actionblock789",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Build Block"
                            },
                            "style": "primary",
                            "url": "https://api.slack.com/tools/block-kit-builder?mode=message&blocks=%5B%5D"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Instructions"
                            },
                            "url": "https://api.slack.com/block-kit"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Example"
                            },
                            "url": "https://api.slack.com/docs/message-buttons"
                        }
                    ]
                }
            ]
        )

        web_client.files_upload(
            channels=channel_id,
            icon_emoji=':robot_face:',
            thread_ts=thread_ts,
            file="HelloWorld.txt",
            title="Test upload"
        )

os.environ['SLACK_TOKEN'] = 'xoxb-827303286006-827334288326-HZYKstrHbjb2DhqQNUDjSyCK'
slack_token = os.environ.get('SLACK_TOKEN')
rtm_client = slack.RTMClient(token=slack_token)
rtm_client.start()