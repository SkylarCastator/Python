import robin_stocks as r
import os
import datetime
import time as t

'''
This is an example script that will print out options data every 10 seconds for 1 minute.
It also saves the data to a txt file. The txt file is saved in the same directory as this code.
'''

#!!! Fill out username and password
username = 'aerialchemist@gmail.com'
password = '3327Aq!3De'
#!!!

login = r.login(username,password)

#!!! fill out the specific option information
strike = 150.0
date = "2019-06-21"
stock = "AAPL"
optionType = "call" #or "put"
#!!!

# File saving variables
minutesToTrack = 1 #in minutes
PrintInterval = 10 #in seconds
endTime = t.time() + 60 * minutesToTrack
fileName = "options.txt"
writeType = "w" #or enter "a" to have it continuously append every time script is run
#

while t.time() < endTime:
    time = str(datetime.datetime.now())
    #Both write and print the data so that you can view it as it runs.
    print(time)
    #Get the data
    instrument_Data = r.get_option_instrument_data(stock,date,strike,optionType)
    market_Data = r.get_option_market_data(stock,date,strike,optionType)

    print("{} Instrument Data {}".format("="*30,"="*30))
    #Instrument_Data is a dictionary, and the key/value pairs can be accessed with .items()
    for key, value in instrument_Data.items():
        print("key: {:<25} value: {}".format(key,value))

    print("{} Market Data {}".format("="*30,"="*30))

    for key, value in market_Data.items():
        print("key: {:<25} value: {}".format(key,value))

    t.sleep(PrintInterval)

#make sure to close the file stream when you are done with it.
fileStream.close()
