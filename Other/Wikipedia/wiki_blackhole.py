import wikipedia
import random

def search_blackhole(init_search, iterations):
    search_page = wikipedia.page(init_search)
    new_search = search_page.links[random.randint(0, len(search_page.links))]
    print(new_search)
    if iterations > 0 :
        remaining_search = iterations-1
        search_blackhole(new_search, remaining_search)
    else:
        print(wikipedia.summary(new_search))

print('Seach Wiki for subject:')
searched_content = input()
#print(searched_content)
#print(wikipedia.summary(searched_content))
search_blackhole(searched_content, 20)
