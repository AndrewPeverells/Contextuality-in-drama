def ArchaismsHandler(text):

    for i,word in enumerate(text): 
        #archaisms
        if (word.endswith('om')):                   
            text[i] = word[:-2]+"um"
        elif (word.endswith("umum", -4)):
            text[i] = word[:-4]+"imum"
        elif (word.endswith("ai", -2)):
            text[i] = word[:-2]+"ae"
    return text         


#todo: *ont-*unt, med-me, *ostr*-*estr*, uelt-uult, *umus-*imus/*ume-*ime/*uma*-*ima*, *oncul*-*uncul*, *ube-*ibe, *issum*-*issim*, quoi*-cui*, acherun*-acheron*