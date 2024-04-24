import re

def replacerfunction(text):
    replacers_always = {
        'v':'u',
        'V':'U',
        'j':'i',
        'J':'I',
        'uors':'uers',
	'uort':'uert',
	'uolt':'uelt',
        }
    
    replacers_standalone = {
        ',':' ',
        '-':'',
        'I.': '',
        'II.': '',
        'III.': '',
        'IV.': '',
        'V.': '',
        'I': '',
        'II': '',
        'III': '',
        'IV': '',
        'V': '',
        'ii': '',
        'iii': '',
        'iv': '',
        'v': '',
        'vi': '',
        'vii': '',
    }

    replaced_text_finale = ""
    replaced_text = re.sub("[^a-zA-Z]+", " ", text)
    for key in replacers_always.keys():
        replaced_text = replaced_text.replace(key,replacers_always[key])
    for word in replaced_text.split():
        wordlenght = len(word)
        for key in replacers_standalone.keys():
            if (wordlenght>len(key) or wordlenght<len(key)):
                continue
            elif(wordlenght==len(key) and word==key):
                word = word.replace(word, replacers_standalone[key])
                replaced_text_finale+=word+" "
                break
        replaced_text_finale+=word+" "
    
    return replaced_text_finale

    