import csv
import numpy as np

#with open('./x06Simple.csv', newline='') as csvfile:
#    data = csv.reader(csvfile, delimiter=',')

#    for row in data:
#        print (row)

#print("\n")

test = np.genfromtxt('./x06Simple.csv', delimiter=',', dtype="uint16", skip_header=1, usecols=(1,2,3))

#print("test", test)
