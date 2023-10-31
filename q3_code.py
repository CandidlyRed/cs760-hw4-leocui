import math
import numpy as np
import pandas as pd


S = [chr(i) for i in range(ord('a'), ord('z') + 1)]
S.append(" ")

class CharNaiveBayes:

    def __init__(self, label, char, fileRange, a = 0.5):
        self.label = label
        self.char = char
        self.fileRange = fileRange
        self.a = a
        self.counts = self.count()
        self.probabilities = self.calcProbabilities()

    def printCharTable(self, type, n=4):
        if type == 1:
            values = self.counts
        elif type == 2:
            values = self.probabilities
        for i in range(len(self.char)):
            print(self.char[i], round(values[i], n))

    def count(self):
        c = np.zeros(len(self.char))
        for i in range(len(self.fileRange)):
            file = open("./hw4Data/" + self.label + str(self.fileRange[i]) + ".txt")
            data = file.read()
            for j in range(len(self.char)):
                c[j] = c[j] + data.count(self.char[j])
        return(c)
    
    def calcProbabilities(self):
        N = sum(self.counts)
        ret = []
        for i in range(len(self.char)):
            p = (self.counts[i] + self.a) / (N + 27 * self.a)
            ret.append(p)
        return(ret)
    
    def logLikelihood(self, fname):
        X = np.zeros(len(self.char))
        data = open("./hw4Data/" + fname + ".txt").read()
        for c in range(len(self.char)):
            X[c] = data.count(self.char[c])
        P = self.probabilities
        return(sum([x * math.log(p) for p, x in zip(P, X)]))

# q2
# print('e')
# c = CharNaiveBayes('e', S, range(10))
# c.printCharTable(2)
# print('j')
# c = CharNaiveBayes('j', S, range(10))
# c.printCharTable(2)
# print('s')
# c = CharNaiveBayes('s', S, range(10))
# c.printCharTable(2)

# q4
# e = CharNaiveBayes("e", S, [10])
# e.printCharTable(1)



# q5
# p = []
# print('e')
# c = CharNaiveBayes('e', S, range(10))
# p.append(c.logLikelihood("e10"))
# p = []
# print('j')
# c = CharNaiveBayes('j', S, range(10))
# p.append(c.logLikelihood("e10"))
# p = []
# print('s')
# c = CharNaiveBayes('s', S, range(10))
# p.append(c.logLikelihood("e10"))

# q7
# m = []

# m.append(CharNaiveBayes('e', S, range(10)))
# m.append(CharNaiveBayes('j', S, range(10)))
# m.append(CharNaiveBayes('s', S, range(10)))

Y = ['e', 'j', 's']
# print("predicted", "actual")
# for l in range(len(Y)):
#     for f in range(10, 20):
#         name = Y[l] + str(f)
#         logLikelihoodArr = []
#         for m in range(len(m)):
#             logLikelihoodArr.append(m[m].logLikelihood(name))
#         maxIndex = np.argmax(logLikelihoodArr)
#         print(Y[maxIndex], Y[l])
