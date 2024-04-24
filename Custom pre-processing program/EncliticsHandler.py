import os

def EncliticsHandler(textlist):
    
    #enclitics removal choice
    encliticsremoval = None
    parolanoneccezione = False

    var = input("Would you like to remove the Enclitics of the text? Please type yes or no: ")
    if (var == "yes"):
        encliticsremoval = True

    enclitics = ["que", "n", "ne", "ue", "ve", "st"] 

    clean_sw = []

    if(encliticsremoval == True):
        for word in textlist:
            if (word == ""):
                continue
            if word.endswith("n",-1):
                    clean_sw += [word[: -len("n")]] + ["ne"]
                    continue
            elif word.endswith("st",-2):
                if word.endswith("ust"):
                    clean_sw += [word[: -len("st") + 1]] + [
                        "est"
                    ]
                    continue
                else:
                    clean_sw += [word[: -len("st")]] + ["est"]
                    continue
            letterainiziale = word[0]
            for enclitic in enclitics:
                    if word.endswith(enclitic)==False:
                        pass
                    else:
                        filename = str('./cartella'+enclitic+'exceptions/'+enclitic+'lettera' + str(letterainiziale) + '.txt')
                        with open(filename, "r") as fileencliticalettera:
                            if (os.stat(filename).st_size!=0):
                                for line in fileencliticalettera:
                                    line = line.rstrip('\n')
                                    if (word==line):
                                        parolanoneccezione=False #word is an exception, do not alter
                                        break
                                    else:
                                        parolanoneccezione=True #word is an exception, cut the enclitic
                                if parolanoneccezione==True:
                                    word2 = word.replace(enclitic,"")
                                    clean_sw.append(word2)
                                    break
                                else:
                                    clean_sw.append(word)
                                    break
                            else:
                                word2 = word.replace(enclitic,"")
                                clean_sw.append(word2)
                                break
                            
            if word.endswith(enclitic)==False:
                clean_sw.append(word)
    else:
        for word in textlist:
            if (word == ""):
                continue
            if word.endswith("n",-1):
                    clean_sw += [word[: -len("n")]] + ["-ne"]
                    continue
            elif word.endswith("st",-2):
                if word.endswith("ust"):
                    clean_sw += [word[: -len("st") + 1]] + [
                        "est"
                    ]
                    continue
                else:
                    clean_sw += [word[: -len("st")]] + ["est"]
                    continue
            letterainiziale = word[0]
            for enclitic in enclitics:
                    if word.endswith(enclitic)==False:
                        pass
                    else:
                        filename = str('./cartella'+enclitic+'exceptions/'+enclitic+'lettera' + str(letterainiziale) + '.txt')
                        with open(filename, "r") as fileencliticalettera:
                            if (os.stat(filename).st_size!=0):
                                for line in fileencliticalettera:
                                    line = line.rstrip('\n')
                                    if (word==line):
                                        parolanoneccezione=False #word is an exception, do not alter
                                        break
                                    else:
                                        parolanoneccezione=True #word is an exception, cut the enclitic
                                if parolanoneccezione==True:
                                    word2 = word.replace(enclitic,"")
                                    clean_sw.append(word2)
                                    clean_sw.append(enclitic)
                                else:
                                    clean_sw.append(word)
                                    break
                            else:
                                word2 = word.replace(enclitic,"")
                                clean_sw.append(word2)
                                clean_sw.append(enclitic)
                                break
            if word.endswith(enclitic)==False:
                clean_sw.append(word)

    return clean_sw