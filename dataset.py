import csv
import random

csvfile = open('Iris.csv', 'r', newline='')
spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
data = []
for row in spamreader:
    data.append(row[0].split(','))
labels = data[0]
data = data[1:]
csvfile.close()
data2 = []
for lst in data:
    if lst[5] == "Iris-setosa":
        lst[5] = '0'
    elif lst[5] == "Iris-versicolor":
        lst[5] = '1'
    else:
        lst[5] = '2'
    data2.append(lst[1:])

data_train = []
data_test = []
data_validation = []
for i in range(150):
    if i < 35 or 49<i<85 or 99<i<135:
        data_train.append(data2[i])
    elif 34<i<45 or 84<i<95 or 134<i<145:
        data_test.append(data2[i])
    else:
        data_validation.append(data2[i])
print(len(data_train))
print(len(data_test))
print(len(data_validation))
random.shuffle(data_train)
random.shuffle(data_test)
random.shuffle(data_validation)


f = open('train.txt', 'w')
for i in data_train:
    s = ''
    for j in i:
        s +=j
        s +=','
    s = s[:-1]
    s += '\n'
    f.write(s)
f.close()

f = open('test.txt', 'w')
for i in data_test:
    s = ''
    for j in i:
        s +=j
        s +=','
    s = s[:-1]
    s += '\n'
    f.write(s)
f.close()

f = open('validation.txt', 'w')
for i in data_validation:
    s = ''
    for j in i:
        s +=j
        s +=','
    s = s[:-1]
    s += '\n'
    f.write(s)
f.close()