from flask import Flask, request
from twilio import twiml


app = Flask(__name__)


@app.route('/sms', methods=['POST'])
def sms():
    number = request.form['From']
    message_body = request.form['Body']

    resp = twiml.Response()
    resp.message('Hello {}, you said: {}'.format(number, message_body))
    return str(resp)

if __name__ == '__main__':
    app.run()

#message = client.messages \
#    .create(
#         from_='whatsapp:+15005550006',
#         body='Hi Joe! Thanks for placing an order with us. We’ll let you know once your order has been processed and delivered. Your order number is O12235234',
#         to='whatsapp:+14155238886'
#     )

#call = client.calls.create(
#                        url='http://demo.twilio.com/docs/voice.xml',
#                        to='+15558675310',
#                        from_='+15017122661'
#                    )

#Run on terminal
#./ngrok http 5000

#http://your-ngrok-url.ngrok.io/sms
#https://www.twilio.com/docs/autopilot/guides/how-to-build-a-chatbot
#https://www.twilio.com/blog/2014/09/burning-man-ticket-notifier-with-twilio-sms-nt.html
