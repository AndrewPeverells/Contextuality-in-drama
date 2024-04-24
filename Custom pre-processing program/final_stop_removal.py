from removeStopwords import removeStopwords

with open('./temp.txt', 'r') as file:
    text = file.read()

# Scelte
stopwordschoice = None

var = input("Would you like to remove Stopwords? Please type yes or no: ")
if (var == "yes"):
    stopwordschoice = True

# Stopwords

if (stopwordschoice == True):
    textlist = removeStopwords(text.split(" "))

text = ""

for element in textlist:
        text += element+" "

with open('./output.txt', 'w') as f:
    f.write(text)