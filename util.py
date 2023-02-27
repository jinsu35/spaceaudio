import re
import os
from datetime import date

CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def log(message):
  envName = os.environ['CONDA_DEFAULT_ENV']
  logDir = '/Users/jinsukim/workspace/whisper/log'
  if not os.path.exists(logDir):
    os.mkdir(logDir)
  today = date.today()
  d4 = today.strftime("%b-%d-%Y")
  filename = f'[{envName}] {d4}.txt'
  path = os.path.join(logDir, filename)
  with open(path, 'a', encoding='utf-8') as f:
    f.write(message)
    f.write('\n')
    f.close()
