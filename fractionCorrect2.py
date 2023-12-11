# To import: from fractionCorrect2 import *
# To run: print(fractionCorrect2(states, seqstr))

def fractionCorrect2(states, seqstr):
    numCorrect = 0
    total = 0
    for i in range (len(states)):
        total += 1
        if (seqstr[i].islower() and states[i] == 'I'):
            numCorrect += 1
        elif (seqstr[i].isupper() and 'E' in states[i]):
            numCorrect += 1
    return (f'Percent Correct: ({(numCorrect/total)*100}%)')

# states = ['I', 'I', 'E1', 'E2', 'E3', 'I', 'E1', 'E2', 'E3', 'I']
# seqstr1 = 'actAGTCGAt'
# seqstr2 = 'acAGTaAGTa'

# print(fractionCorrect2(states, seqstr2))
