import time
from pytrends.request import TrendReq
import pandas as pd

pytrends = TrendReq(hl='en-US', tz=360)
keywords = ["Air Fryer", "Smartwatch"]
pytrends.build_payload(keywords, cat=0, timeframe='today 3-m', geo='', gprop='')

try:
    data = pytrends.interest_over_time()
    print(data)
except Exception as e:
    print(e)

time.sleep(30)