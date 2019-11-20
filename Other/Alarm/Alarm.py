import time
import webbrowser
import random
import Podcast as podcast

def start_alarm():
    print ('what time do you want to wake up\?')
    print ('Use this form: \nExample: 06:30:00')
    alarm_time = set_alarm("> ")
    current_time = time.strftime("%H:%M:%S")
    print ("Starting Alarm")
    while time != alarm_time:
        print ("The time is " + current_time)
        current_time = time.strftime("%H:%M:%S")
        time.sleep(1)
        if current_time == alarm_time:
                activate_alarm()

def set_alarm(prompt):
    value = input(prompt)
    try:
        time.strptime(value[:8], '%H:%M:%S')
        return value
    except ValueError:
        print("Sorry, That wasn't a correct time.")
    return set_alarm(prompt)

def activate_alarm():
    print("Time to wake up!")
    podcast.play_podcast()

start_alarm()
