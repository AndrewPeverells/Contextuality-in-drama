import string

n_exceptions = [
    "aenean",
    "agmen",
    "alioquin",
    "an",
    "attamen",
    "cacumen",
    "carmen",
    "certamen",
    "clymenen",
    "cognomen",
    "crimen",
    "culmen",
    "dein",
    "deucalion",
    "discrimen",
    "en",
    "epitheton",
    "erinyn",
    "exin",
    "flumen",
    "forsan",
    "forsitan",
    "fulmen",
    "gramen",
    "hymen",
    "iason",
    "in",
    "limen",
    "liquamen",
    "lumen",
    "nomen",
    "non",
    "numen",
    "omen",
    "orion",
    "paean",
    "pan",
    "pelion",
    "phaethon",
    "python",
    "quin",
    "semen",
    "sin",
    "specimen",
    "tamen",
    "themin",
    "titan",
]

nexceptions_dict = {  
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
    for word in n_exceptions:
        if word[0] == letter:
            listaparoleperlettera.append(word)
    print(listaparoleperlettera)
    nexceptions_dict[letter].extend(listaparoleperlettera)
    listaparoleperlettera=listaparoleperlettera.clear()
    listaparoleperlettera = []

print(nexceptions_dict)

for key, value in nexceptions_dict.items():
  filename = str('./cartellanexceptions/nlettera' + str(key) + '.txt')
  with open(filename, "w") as f:
    for v in value:
      f.write(v + "\n")