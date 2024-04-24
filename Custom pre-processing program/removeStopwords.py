from nltk.tokenize.punkt import PunktLanguageVars
import re

def removeStopwords(clean_text):
    for w in clean_text:
        if (w == ""):
            continue
        letterainiziale = w[0]
        filename = str('./cartellastopwords/lettera' + str(letterainiziale) + '.txt')
        with open(filename, "r") as filestopwordslettera:
            for line in filestopwordslettera:
                line = line.rstrip('\n')
                if (w==line):
                    clean_text.remove(w)
                    break
    return clean_text