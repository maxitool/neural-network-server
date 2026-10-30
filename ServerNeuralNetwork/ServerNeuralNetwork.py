import socket
import json
import ctypes
from telnetlib import DO
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

CONST_IMAGE_HEIGHT = 64
CONST_IMAGE_WIDTH = 64
CONST_LENGTH_IMAGE = CONST_IMAGE_HEIGHT * CONST_IMAGE_WIDTH * 3  # 3 - length_value
CONST_SIZE_STRING_VALUE = 2
CONST_SETTINGS_PATH = 'settings.json'
CONST_ALL_KANJI_PATH = 'ALL_CLASS_KANJI.txt'
CONST_COUNT_GROUPS = 54

def read_settings(filepath) -> json:
    with open(filepath, 'r') as settingsFile:
        return json.loads(settingsFile.read())

def read_all_groups_kanji(filepath):
    groups = []
    current_class = 0
    with open("ALL_CLASS_KANJI.txt", encoding="utf8") as f:
        for line in f:
            groups.append(line[len(str(current_class)):])
            current_class = current_class + 1
    for i in range(len(groups)):
        groups[i] = groups[i][:- 1]
    f.close()
    return groups

def Write_Data_In_File(data):
    with open('data.txt','w') as f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                f.write(str(data[i][j]) + ' ')
            f.write('\n')

def vector_to_array(vector):
    array = np.zeros(shape=(CONST_IMAGE_HEIGHT, CONST_IMAGE_WIDTH))
    for i in range(CONST_IMAGE_HEIGHT):
        for j in range(CONST_IMAGE_WIDTH):
            array[i][j] = vector[0][i][j]
    Write_Data_In_File(array)
    array = np.expand_dims(array, axis=0)
    array = np.expand_dims(array, axis=3)
    return array

def my_sort(data):
    result = []
    max = 0
    countClasses = len(data)
    numberData = 0
    while countClasses > 0:
        for i in range(len(data)):
            if data[i][1] > max:
                max = data[i][1]
                numberData = i
        result.append(data[numberData][0])
        data[numberData][1] = 0
        max = 0
        countClasses = countClasses - 1
    return result

def get_results_NN(min_value, result):
    data = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j] >= min_value:
                data.append([j, result[i][j]])
    data = my_sort(data)
    return data

def secondary_classifier_invoke(group_number, image):
    neural_network = model_create(len(kanji_groups[group_number]))
    neural_network.load_weights('Weights/' + str(group_number) + '.h5')
    classes = neural_network.predict(image)
    classes = get_results_NN(0.5, classes) # 0.5 - min value of prediction probability
    if len(classes) == 0:
        return '?'
    return kanji_groups[group_number][classes[0]]

def neural_network_system_invoke(image):
    list_kanji = ''
    #The main classifier predicts the groups to which this hieroglyph may belong, 0.2 - min value of prediction probability
    groups = main_neural_network.predict(image)
    groups = get_results_NN(0.2, groups)
    if len(groups) == 0:
        return '?'
    #The secondary classifier predicts hieroglyph
    for i in range(len(groups)):
        if groups[i] == 53:
            list_kanji = list_kanji + '一'
            continue
        kanji = secondary_classifier_invoke(groups[i], image)
        if kanji != '?':
            list_kanji = list_kanji + kanji
    if len(list_kanji) == 0:
        return '?'
    return list_kanji


def model_create(count_classes):
    model = keras.Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,1)),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=2),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D((2,2), strides=2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(count_classes, activation='softmax')
        ])
    return model

#Add .so library
file = ctypes.CDLL('./Decoding.so')
file.Decoding.restype = ctypes.POINTER(ctypes.c_double * CONST_IMAGE_WIDTH * CONST_IMAGE_HEIGHT)
file.Decoding.argtypes = [ctypes.c_char_p]
#Read settings
settings = read_settings(CONST_SETTINGS_PATH)
print(settings)
#Read all kanji groups
kanji_groups = read_all_groups_kanji(CONST_ALL_KANJI_PATH)
#Initialization of the main neural network
main_neural_network = model_create(CONST_COUNT_GROUPS)
main_neural_network.load_weights('Weights/Main.h5')
#Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((settings["host"], settings["port"]))
server_socket.listen(1) #Queue is 1
print(settings["name"] + ' started\n')
while True:
    conn, addr = server_socket.accept()
    print("Сonnection established")
    encoded_image = ''
    while True:
        data = conn.recv(CONST_LENGTH_IMAGE * CONST_SIZE_STRING_VALUE)
        encoded_image = encoded_image + data.decode("utf-8")
        if encoded_image[len(encoded_image) - 2] == 'w':
            break
    encoded_image = encoded_image[:len(encoded_image) - 2]
    encoded_image = encoded_image.encode('utf-8')
    if data != None:
        if (len(encoded_image) == CONST_LENGTH_IMAGE):
            image = file.Decoding(encoded_image)
            image = vector_to_array(image)
            list_kanji = neural_network_system_invoke(image)
            conn.sendall(list_kanji.encode("utf-8"))
            data = conn.recv(1)
            conn.close()
            print("Сonnection closed\n")
        else:
            print("Data length is small or big: " + str(len(encoded_image)))
            conn.close()
    else:
        print("Data is empty")
        conn.close()
