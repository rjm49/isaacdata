from random import randint

import numpy
from keras import Sequential
from keras.layers import Dense, Activation

base = "../../../isaac_data_files/"

#need to build a softmax classifier to recommend questions...
#this would have qn output nodes

#get assignments ....
ass_df = open(base+"gb_assignments.csv")
gbd_df = None #we need to get the gameboards in here

#then snapshot the student's profile against the questions that were set for them...
#using one hot encoding....
nin = 132
nout = 6060

model = Sequential([
    Dense(nin, input_dim=nin),
    Activation('relu'),
    Dense(nin),
    Activation('relu'),
    Dense(nout),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

inputs = numpy.ndarray(shape=nin)
outputs = numpy.ndarray(shape=nout)
for _ in range(1000):
    input_stuff = numpy.random.randint(0,1,nin).reshape((-1,nin))
    output_stuff = numpy.random.randint(0,1,nout).reshape((-1,nout))
    inputs = numpy.vstack((inputs, input_stuff))
    outputs = numpy.vstack((outputs, output_stuff))


model.fit(inputs, outputs)

for _ in range(10):
    input_stuff = numpy.random.randint(0,1,nin).reshape((-1,nin))
    predictions = model.predict(input_stuff)
    for p in predictions:
        print(p)