"""
api.giphy.com/v1/gifs/search?api_key=dc6zaTOxFJmzC&&q=cheese&&limit=2&&offset0&&rating=g&&lang=eng&&fmt=json
"""
import time
import giphy_client
from giphy_client.rest import ApiException
import ast
import random

# create an instance of the API class
api_instance = giphy_client.DefaultApi()
api_key = 'dc6zaTOxFJmzC' # str | Giphy API Key.
q = 'sexy' # str | Search query term or prhase.
limit = 25 # int | The maximum number of records to return. (optional) (default to 25)
offset = 0 # int | An optional results offset. Defaults to 0. (optional) (default to 0)
rating = 'r' # str | Filters results by specified rating. (optional)
lang = 'en' # str | Specify default country for regional content; use a 2-letter ISO 639-1 country code. See list of supported languages <a href = \"../language-support\">here</a>. (optional)
fmt = 'json' # str | Used to indicate the expected response format. Default is Json. (optional) (default to json)

try: 
    # Search Endpoint
    api_response = api_instance.gifs_search_get(api_key, q, limit=limit, offset=offset, rating=rating, lang=lang, fmt=fmt)
    repository = api_response.data[random.randrange(0, 25, 1)]
    print(repository.images.downsized.url)
except ApiException as e:
    print("Exception when calling DefaultApi->gifs_search_get: %s\n" % e)
