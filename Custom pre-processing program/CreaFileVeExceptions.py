import string

ve_exceptions = [
    "agave",
    "ave",
    "bove",
    "breve",
    "calve",
    "cave",
    "cive",
    "curve",
    "fave",
    "furtive",
    "gradive",
    "grave",
    "ignave",
    "iove",
    "lascive",
    "leve",
    "move",
    "nave",
    "neve",
    "nive",
    "praegrave",
    "promiscue",
    "prospicve",
    "proterve",
    "remove",
    "resolve",
    "saeve",
    "salve",
    "sive",
    "solve",
    "summove",
    "vive",
    "vove",
]

veexceptions_dict = {  
    'a' : [],
    'b' : [],
    'c' : [],
    'd' : [],
    'e' : [],
    'f' : [],
    'g' : [],
    'h' : [],
    'i' : [],
    'j' : [],
    'k' : [],
    'l' : [],
    'm' : [],
    'n' : [],
    'o' : [],
    'p' : [],
    'q' : [],
    'r' : [],
    's' : [],
    't' : [],
    'u' : [],
    'v' : [],
    'w' : [],
    'x' : [],
    'y' : [],
    'z' : []
}
listaparoleperlettera = []


for letter in string.ascii_lowercase:
    print(listaparoleperlettera)
    for word in ve_exceptions:
        if word[0] == letter:
            listaparoleperlettera.append(word)
    print(listaparoleperlettera)
    veexceptions_dict[letter].extend(listaparoleperlettera)
    listaparoleperlettera=listaparoleperlettera.clear()
    listaparoleperlettera = []

print(veexceptions_dict)

for key, value in veexceptions_dict.items():
  filename = str('./cartellaveexceptions/velettera' + str(key) + '.txt')
  with open(filename, "w") as f:
    for v in value:
      f.write(v + "\n")