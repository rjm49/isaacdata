import keras
import numpy
from keras import Sequential, Input, Model
from keras.layers import Dropout, Dense, Concatenate, concatenate, LeakyReLU, BatchNormalization, Reshape, activations, \
    Embedding, LSTM, regularizers
from keras.optimizers import Adam, Adadelta, rmsprop

# def edit_distance(true_y, pred_y):
#     pred_y = K.eval(pred_y)
#     true_y = K.eval(true_y)
#     card = numpy.sum(true_y) # get cardinality
#     yixs = numpy.argsort(pred_y)
#
#     K.in_top_k(pred_y, true_y)
#     yixs = list(reversed(yixs))[0:card]
#     pred_y = numpy.zeros(len(pred_y))
#     pred_y[yixs] = 1.0
#     score = numpy.sum(numpy.logical_or(pred_y, true_y)) - numpy.sum(numpy.logical_and(pred_y, true_y))
#     return score



def make_mixed_loss_model(n_S, n_Q, n_P, n_C, n_T, n_L, n_V):

    # this is our input placeholder
    input_S = Input(shape=(n_S,), name="semistatic_input")
    input_Q = Input(shape=(n_Q,), name="wide_input")

    # "encoded" is the encoded representation of the input
    # encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # "decoded" is the lossy reconstruction of the input
    hwidth = 1.1*(n_S+n_Q)

    # hidden = Dense(8000, activation='relu')(input_Q)
    # hidden = Dense(6000, activation='relu')(concatenate([input_S, hidden]))
    # hidden = Dense(4000, activation='relu')(hidden)

    hidden = Dense(1000, activation='relu')(input_Q)
    hidden = Dense(750, activation='relu')(concatenate([input_S, hidden]))
    hidden = Dense(375, activation='relu')(hidden)


    next_concepts = Dense(n_C, activation='sigmoid', name="next_concepts")(hidden)
    next_topic = Dense(n_T, activation='sigmoid', name="next_topic")(hidden)
    next_level = Dense(n_L, activation='relu', name="next_level")(hidden)
    next_vol = Dense(n_V, activation='relu', name="next_vol")(hidden)

    # hidden = Dense(256, activation='relu')(concatenate([hidden, next_concepts, next_topic, next_level, next_vol]))
    next_qns = Dense(n_P, activation='sigmoid', name="next_qn")(hidden)

    # hidden = LeakyReLU(1024)(concatenate([input_S, input_Q]))
    # hidden = LeakyReLU(64)(hidden)
    # hidden = LeakyReLU(256)(hidden)

    # next_qns = Dense(n_P, activation='sigmoid', name="next_qn")(hidden_2)


    # next_concept = Dense(n_C, activation='softmax', name="next_concepts")(encoded)
    # next_level = Dense(n_L, activation='relu', name="next_level")(encoded)

    # this model maps an input to its reconstruction
    model = Model(inputs=[input_S, input_Q], outputs=(next_qns, next_concepts, next_topic, next_level, next_vol) )
    model.compile(optimizer="rmsprop", loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','mae','mae'],  loss_weights = [1,1,1,1,1]) #[500,100,1,5,50])
    model.summary()
    # input("press RETURN to CONTINUE")
    # model = Model(inputs=[input_layer], outputs=[decoded] )
    # model.compile(optimizer='adam', loss=['categorical_crossentropy'],
    #               loss_weights=[1.0], metrics=['accuracy','mse'])
    return model

def build_workload_model(n_S,n_Q,n_L,n_V):
    # this is our input placeholder
    input_S = Input(shape=(n_S,), name="semistatic_input")
    input_Q = Input(shape=(n_Q,), name="wide_input")
    hidden_1 = Dense(256, activation='relu')(concatenate([input_S, input_Q]))
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    next_level = Dense(n_L, activation='relu', name="next_level")(hidden_2)
    next_vol = Dense(n_V, activation='relu', name="next_vol")(hidden_2)
    m = Model(inputs=[input_S, input_Q], outputs=[next_level, next_vol] )
    m.compile(optimizer="rmsprop", loss=['mse', 'mse'],  loss_weights = [1,1])
    return m


def make_adversarial_pair(w_profile, n_P):
    #build generator
    g_w = 128
    g_hidden = Input(shape=(w_profile,), name="g_input")
    g_hidden = LeakyReLU(g_w)(g_hidden)
    g_hidden = LeakyReLU(g_w)(g_hidden)
    next_pg = Dense(n_P, activation='softmax', name="next_pg")(g_hidden)

    #build discriminator
    d_in = Input(n_P)
    q_in = Input(n_Q)
    d_hidden = concatenate([d_in, q_in])
    d_out= Dense(activation='sigmoid')


def make_model(n_in, n_out):
    print(n_in, n_out)
    mode="ML_SOFTMAX_DOUT"

    model = Sequential()

    if mode=="MLBIN":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(4800, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(2400, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    elif mode=="MLBIN_SMALL":
        # model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(n_out, activation='relu', input_dim=n_in))
        # model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(8*n_out, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(8*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    elif mode=="ML_SOFTMAX":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(8*n_out, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(4*n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2*n_out, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(n_out, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adadelta()
        model.compile(loss="categorical_crossentropy", optimizer=optimiser)
    elif mode=="ML_SOFTMAX_2H":
        model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(2*n_in, activation='relu', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(int(n_out + n_in), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(int(0.5*(n_out + n_in)), activation='relu'))
        model.add(Dense(n_out, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adadelta()
        model.compile(loss="binary_crossentropy", optimizer=optimiser)
    elif mode=="ML_SOFTMAX_WIDE":
        # model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(2*n_in, activation='tanh', input_dim=n_in))
        # model.add(Dropout(0.5))
        # model.add(Dense(n_in, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(n_in, activation='tanh'))
        # model.add(Dropout(0.5))
        # model.add(Dense(n_out, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='sigmoid'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # optimiser = Adam(0.0002, 0.5)
        optimiser = Adam()
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    elif mode=="ML_SOFTMAX_DOUT":
        # model.add(Dropout(0.5, input_shape=(n_in,)))
        model.add(Dense(n_in, activation='tanh', input_dim=n_in))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='tanh'))
        # model.add(Dropout(0.5))
        model.add(Dense(n_out, activation='relu'))
        model.add(Dense(n_out, activation='softmax'))
        model.compile(loss="binary_crossentropy", optimizer='rmsprop')
    return model

def LSTM_model(n_S, n_Q, n_P, n_C, n_T, n_L, n_V):
    # main_input = Input(shape=(None, n_Q), dtype='int32', name='main_input')
    # This embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    input = Input(shape = (100,), dtype='int32')
    embd_Q = Embedding(input_dim=n_Q, output_dim=256, input_length=None)(input)
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    q_hist_encoded = LSTM(256)(embd_Q)
    input_S = Input(shape=(n_S,), name="semistatic_input")

    hidden = Dense(256, activation='relu')(concatenate([input_S, q_hist_encoded]))
    hidden = Dense(256, activation='relu')(hidden)
    hidden = Dense(256, activation='relu')(hidden)

    next_concepts = Dense(n_C, activation='sigmoid', name="next_concepts")(hidden)
    next_topic = Dense(n_T, activation='sigmoid', name="next_topic")(hidden)
    next_level = Dense(n_L, activation='relu', name="next_level")(hidden)
    next_vol = Dense(n_V, activation='relu', name="next_vol")(hidden)

    # hidden = Dense(256, activation='relu')(concatenate([hidden, next_concepts, next_topic, next_level, next_vol]))
    next_qns = Dense(n_P, activation='sigmoid', name="next_qn")(hidden)

    model = Model(inputs=[input_S, embd_Q], outputs=(next_qns, next_concepts, next_topic, next_level, next_vol) )
    model.compile(optimizer="rmsprop", loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','mae','mae'],  loss_weights = [1,1,1,1,1]) #[500,100,1,5,50])
    model.summary()
    return model