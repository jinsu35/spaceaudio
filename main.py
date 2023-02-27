import sys, getopt
import subprocess

from translate import writeTranscript 
from speech import makeSpeech

supportedLanguages = [
   'arabic',
   'czech',
   'german',
   'spanish',
   'estonian',
   'finnish',
   'french',
   'italian',
   'japanese',
   'lithuanian',
   'latvian',
   'dutch',
   'romanian',
   'turkish',
]

def main(argv):
   name = ''
   output = ''
   subtitlePath = ''
   language = ''

   opts, args = getopt.getopt(argv,"hn:p:l:o:",["name=", "pathToSubtitle=", "lang=", "output="])
   if len(opts) == 0:
      print('Usage: main.py -n <name> -p <pathToSubtitle> -l <language> -o <output>')
      quit()
   for opt, arg in opts:
      if opt == '-h':
         print ('main.py -n <name> -p <pathToSubtitle> -l <language> -o <output>')
         sys.exit()
      elif opt in ("-n", "--name"):
         name = arg
      elif opt in ("-p", "--pathToSubtitle"):
         subtitlePath = arg
      elif opt in ("-l", "--lang"):
         language = arg
      elif opt in ("-o", "--output"):
         output = arg
   if name == '':
      print('<name> must be specified')
      quit()
   if subtitlePath == '':
      print('<pathToSubtitle> must be specified')
      quit()
   if language not in supportedLanguages:
      s = ', '.join(supportedLanguages)
      print(f'<language> must be one of {s}.')
      quit()
   if output == '':
      output = f'{name}_{language}.m4a'

   subprocess.call(['mkdir', '-p', f'media/{name}/subtitle'])
   subprocess.call(['mkdir', '-p', f'media/{name}/audio/{language}'])

   writeTranscript(name, subtitlePath, language)
   makeSpeech(name, language)
   subprocess.call(['cp', f'media/{name}/{name}_{language}.m4a', output])

if __name__ == "__main__":
   main(sys.argv[1:])