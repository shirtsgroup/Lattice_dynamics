import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename")
parser.add_option("-n", dest="towrite")
(options, args) = parser.parse_args()
filename = options.filename
towrite = options.towrite
file1 = open(filename,'r')
filelines = file1.readlines()

numlines = 0
test = False
while test == False:
    end = filelines[numlines]
    totest = end.split()[0]
    if totest == 'END':
        test = True
    numlines +=1

numtowrite = numlines-5+3

f = open(towrite, 'w+')
numtotal = len(filelines)

for x in range(numtotal-numtowrite, numtotal):
    f.write(filelines[x])

f.close()



