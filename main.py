import numpy as np
import math

# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.wh1h2 = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.wh2o = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1+(math.e)**(-x))

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into 1-st hidden layer
        hidden_1_inputs = np.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden layer
        hidden_1_outputs = self.activation_function(hidden_1_inputs)

        # calculate signals into 2-nd hidden layer
        hidden_2_inputs = np.dot(self.wh1h2, hidden_1_outputs)
        # calculate the signals emerging from hidden layer
        hidden_2_outputs = self.activation_function(hidden_2_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.wh2o, hidden_2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_2_errors = np.dot(self.wh2o.T, output_errors)
        hidden_1_errors = np.dot(self.wh1h2.T, hidden_2_errors)

        # update the weights for the links between the hidden and output layers
        self.wh2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_2_outputs))

        self.wh1h2 += self.lr * np.dot((hidden_2_errors * hidden_2_outputs * (1.0 - hidden_2_outputs)),
                                      np.transpose(hidden_1_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih1 += self.lr * np.dot((hidden_1_errors * hidden_1_outputs * (1.0 - hidden_1_outputs)),
                                        np.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into 1-st hidden layer
        hidden_1_inputs = np.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden layer
        hidden_1_outputs = self.activation_function(hidden_1_inputs)

        # calculate signals into 2-nd hidden layer
        hidden_2_inputs = np.dot(self.wh1h2, hidden_1_outputs)
        # calculate the signals emerging from hidden layer
        hidden_2_outputs = self.activation_function(hidden_2_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.wh2o, hidden_2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 4
hidden_1_nodes = 5
hidden_2_nodes = 7
output_nodes = 3

# learning rate is 0.1
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, learning_rate)

'''
# загрузить в список тренировочный набор данных
# CSV-файла набора MNIST
training_data_file = open('train.txt', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# тренировка нейронной сети
# переменная epochs указывает, сколько раз тренировочный
# набор данных используется для тренировки сети
epochs = 7
for e in range(epochs):
    # перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
        # получить список значений из записи, используя символы
        # запятой (',f) в качестве разделителей
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (np.asfarray(all_values[:-1]))
        # создать целевые выходные значения (все равны 0,01, за
        # исключением желаемого маркерного значения, равного 0,99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] - целевое маркерное значение
        # для данной записи
        targets[int(all_values[-1])] =0.99
        n.train(inputs, targets)
        pass
    pass
'''
# Записываем нейросеть из файлов
f = open('wih1.txt', 'r')
s = f.read()
wih_1 = s.split(' /n')
for i in range(5):
    wih_2 = wih_1[i].split(' ')
    for j in range(4):
        n.wih1[i][j] = np.float64(wih_2[j])
f.close()

f = open('wh1h2.txt', 'r')
s = f.read()
wih_1 = s.split(' /n')
for i in range(7):
    wih_2 = wih_1[i].split(' ')
    for j in range(5):
        n.wh1h2[i][j] = np.float64(wih_2[j])
f.close()

f = open('wh2o.txt', 'r')
s = f.read()
who_1 = s.split(' /n')
for i in range(3):
    who_2 = who_1[i].split(' ')
    for j in range(7):
        n.wh2o[i][j] = np.float64(who_2[j])
f.close()




# загрузить в список тестовый набор данных
# CSV-файла набора MNIST
test_data_file = open("validation.txt", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# тестирование нейронной сети
# журнал оценок работы сети, первоначально пустой
scorecard = []
# перебрать все записи в тестовом наборе данных
for record in test_data_list:
    # получить список значений из записи, используя символы
    # запятой (',*) в качестве разделителей
    all_values = record.split(',')
    # правильный ответ - первое значение
    correct_label = int(all_values[-1])
    # масштабировать и сместить входные значения
    inputs = (np.asfarray(all_values[:-1]))
    # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением
    label = np.argmax(outputs)
    # присоединить оценку ответа сети к концу списка
    if (label == correct_label):
        # в случае правильного ответа сети присоединить
        # к списку значение 1
        scorecard.append(1)
    else:
        # в случае неправильного ответа сети присоединить
        # к списку значение 0
        scorecard.append(0)
        pass
    pass

# рассчитать показатель эффективности в виде
# доли правильных ответов
scorecard_array = np.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum() / scorecard_array.size)
print(scorecard_array.sum())
print(scorecard)

'''
f = open('wih1_2.txt', 'w')
for i in range(5):
    s = ''
    for j in n.wih1[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
f.close()

f = open('wh1h2_2.txt', 'w')
for i in range(7):
    s = ''
    for j in n.wh1h2[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
f.close()

f = open('wh2o_2.txt', 'w')
for i in range(3):
    s = ''
    for j in n.wh2o[i]:
        s += str(j)
        s += ' '
    f.write(s)
    f.write('/n')
    '''
