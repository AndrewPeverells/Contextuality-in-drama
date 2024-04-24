PURPOSE AND MODULES

This is a custom written program that addresses the issue of having to deal with orthographically not-unified Latin texts.
The program includes 4 main modules that address different issues.

- General module

  Orthography issues: tokenizer, j > i, v > u, elimination of punctuation and of roman and arabic numerals and non UTF-8 symbols, replacer of every capital letter into non capital.
  
  (Optional)
- Enclitics handler

  Splits the enclitic from the root word. Exceptions are handled through a list of custom written exception words for every main Latin enclitic (que, ne-n, st, ue-ve: CartellaQueExceptions, CartellaNeExceptions, CartellaNExceptions, CartellaStExceptions, CartellaUeExceptions, CartellaVeExceptions).
- Archaisms handler

  Translates archaisms into classical Latin variants.
  Still to do: *ont-*unt, med-me, *ostr*-*estr*, uelt-uult, *umus-*imus/*ume-*ime/*uma*-*ima*, *oncul*-*uncul*, *ube-*ibe, *issum*-*issim*, quoi*-cui*, acherun*-acheron*
- Stopwords handler

  Removes stopwords from a custom built list of Latin stopwords (from the folder "cartellastopwords", divided per letter). You can add your own custom stopwords to the files.

---

Developed and maintained by:

- Andrea Peverelli, junior researcher and PhD candidate in Digital Humanities for the Translatin Project (PI: Jan Bloemendal) at the Huygens Institute, KNAW Humanities Cluster, Amsterdam (Netherlands)
- Alessandro Rossi, Politecnico di Milano, Dipartimento di Ingegneria Informatica (Computer Engineering Department)

The program is completely free and open access. We kindly ask you to cite this GitHub repository and the authors for reference in your own research if you end up using CURRENS.

---

HOW TO USE THE PROGRAM

1. Download the whole folder and store locally.
2. Open the currens.py file in a Python environment editor (VSCode, Sublime, Atom, Anaconda, Jupyter...).
3. At line 10:
```with open('Path/to/your/file.txt', 'r') as file:```
insert the full path of your text file (in txt format).
4. A prompt will appear in your consol asking you to make choices for the three optional modules of CURRENS (archaisms, stopwords and enclitics handler):

![currens tutorial](https://github.com/AndrewPeverells/CURRENS/assets/45845685/56b83004-f920-4b6f-a348-e0929be62fba)

Type yes or no in the console for each, depending on the module you require for your experiment.

5. You should have now a txt file called "temp" in the CURRENS folder. This is the output file of the cleaned text after the process.
