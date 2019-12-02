from time import time
import random
from os import system, name
import time as timer

import requests
import json

app_id = 'ad0d2054'
app_key = '58e86197b57fa6838e93cf48915fce7c'

language = 'en-gb'
fields = 'definitions'
strictMatch = 'false'

i = 0
cs = False
list = []

f = open('words.txt', 'r')
for line in f:
    list.append(line.strip())

random.shuffle(list)

print("---DICTIONARY SPEED TYPING TUTOR---")
timer.sleep(1)
print("Total words in dictionary: ", len(list))
timer.sleep(2)

search_word = list[0]
#print (search_word)
url = 'https://od-api.oxforddictionaries.com:443/api/v2/entries/' + language + '/' + search_word.lower() + '?fields=' + fields + '&strictMatch=' + strictMatch;

r = requests.get(url, headers = {'app_id': app_id, 'app_key': app_key})

#print("code {}\n".format(r.status_code))
content = r.json()
#print("json \n" + content)
definition = content['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['definitions'][0]

prompt = list[0] + " " + definition

def counter():
	i = 0 
	print (prompt)
	input(">> Press ENTER to begin")
	begin_time = time()
	inp = input("\n")
	end_time = time()
	final_time = (end_time - begin_time) / 60
	return final_time, inp


def wpm(time, line):
	words = line.split()
	word_length = len(words)
	words_per_m = word_length / time
	return words_per_m


def wordcheck(inp):
	prompts = prompt.split()
	inputs = inp.split()
	errorcount = 0
	
	idx = 0
	for inp in inputs:
		if inp != prompts[idx]:
			errorcount += 1
			if inp == prompts[idx + 1]:
				idx += 2
			elif inp != prompts[idx - 1]:
				idx += 1
		else:
			idx += 1

	words_left = len(prompts) - len(inputs)
	correct = float(len(prompts)) - float(errorcount)
	percentage = (((float(correct) / float(len(prompts))) - float(words_left) / float(len(prompts))) * 100)

	
	return percentage


tm, line = counter()
tm = round(tm, 2)
words_per_minute = wpm(tm, line)
words_per_minute = round(words_per_minute, 2)
print("You total time was: {0} minutes".format(tm))
print("with an average of: {0} words per minute".format(words_per_minute))
percentage = wordcheck(line)
percentager = round(percentage, 2)
print("with an accuracy of: {0} %% accuracy".format(percentager))
