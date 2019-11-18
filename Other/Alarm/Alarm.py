import time
import webbrowser
import random
import Podcast as podcast

def start_alarm():
    print ('what time do you want to wake up\?')
    print ('Use this form: \nExample: 06:30:00')
    alarm_time = set_alarm("> ")
    current_time = time.strftime("%H:%M:%S")

    while time != alarm_time:
        #print ("The time is " + current_time)
        current_time = time.strftime("%H:%M:%S")
        time.sleep(1)
        if current_time == alarm_time:
                activate_alarm()

def set_alarm(prompt):
    while True:
	value = input(prompt)
        try:
            time.strftime(value, "%H:%M:%S")
        except ValueError:
            print("Sorry, That wasn't a correct time.")
            continue
        break
    return value

def activate_alarm():
    print("Time to wake up!")
    podcast.play_podcast()

start_alarm()
