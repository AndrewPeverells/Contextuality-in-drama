import string

st_exceptions = [
    "abest",
    "adest",
    "ast",
    "deest",
    "est",
    "inest",
    "interest",
    "post",
    "potest",
    "prodest",
    "subest",
    "superest",
]

stexceptions_dict = {  
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
    for word in st_exceptions:
        if word[0] == letter:
            listaparoleperlettera.append(word)
    print(listaparoleperlettera)
    stexceptions_dict[letter].extend(listaparoleperlettera)
    listaparoleperlettera=listaparoleperlettera.clear()
    listaparoleperlettera = []

print(stexceptions_dict)

for key, value in stexceptions_dict.items():
  filename = str('./cartellastexceptions/stlettera' + str(key) + '.txt')
  with open(filename, "w") as f:
    for v in value:
      f.write(v + "\n")