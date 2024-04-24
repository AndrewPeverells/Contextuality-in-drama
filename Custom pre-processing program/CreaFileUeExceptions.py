import string

ue_exceptions = [
    "agaue",
    "ambigue",
    "assidue",
    "aue",
    "boue",
    "breue",
    "calue",
    "caue",
    "ciue",
    "congrue",
    "contigue",
    "continue",
    "curue",
    "exigue",
    "exolue",
    "exue",
    "fatue",
    "faue",
    "fue",
    "furtiue",
    "gradiue",
    "graue",
    "ignaue",
    "incongrue",
    "ingenue",
    "innocue",
    "ioue",
    "lasciue",
    "leue",
    "moue",
    "mutue",
    "naue",
    "neue",
    "niue",
    "perexigue",
    "perspicue",
    "pingue",
    "praecipue",
    "praegraue",
    "prospicue",
    "proterue",
    "remoue",
    "resolue",
    "saeue",
    "salue",
    "siue",
    "solue",
    "strenue",
    "sue",
    "summoue",
    "superflue",
    "supplicue",
    "tenue",
    "uiue",
    "ungue",
    "uoue",
]

ueexceptions_dict = {  
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
    for word in ue_exceptions:
        if word[0] == letter:
            listaparoleperlettera.append(word)
    print(listaparoleperlettera)
    ueexceptions_dict[letter].extend(listaparoleperlettera)
    listaparoleperlettera=listaparoleperlettera.clear()
    listaparoleperlettera = []

print(ueexceptions_dict)

for key, value in ueexceptions_dict.items():
  filename = str('./cartellaueexceptions/uelettera' + str(key) + '.txt')
  with open(filename, "w") as f:
    for v in value:
      f.write(v + "\n")