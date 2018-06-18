from keras import Model, Input
from keras.layers import Dense, concatenate
from keras.optimizers import Adam


def build_pass_model(n_S,n_X,n_U):
    input_S = Input(shape=(n_S,), name="s_input")
    input_X = Input(shape=(n_X,), name="x_input")
    input_U = Input(shape=(n_U,), name="u_input")
    w = 1024
    hidden = Dense(w, activation='relu')(concatenate([input_S, input_X, input_U]))
    out = Dense(1, activation='sigmoid', name="pass_prob")(hidden)
    o = Adam()
    m = Model(inputs=[input_S,input_X, input_U], outputs=out )
    m.compile(optimizer=o, loss='categorical_crossentropy', metrics=['acc'])
    m.summary()
    return m

def build_atts_model(n_S,n_X,n_U):
    input_S = Input(shape=(n_S,), name="s_input")
    input_X = Input(shape=(n_X,), name="x_input")
    input_U = Input(shape=(n_U,), name="u_input")
    w = 1024
    hidden = Dense(w, activation='relu')(concatenate([input_S, input_X, input_U]))
    out = Dense(1, activation='relu', name="num_atts")(hidden)
    o = Adam()
    m = Model(inputs=[input_S,input_X, input_U], outputs=out )
    m.compile(optimizer=o, loss='mse')
    m.summary()
    return m

class ZPDPredictor():
    def __init__(self):
        self.pass_model = None
        self.atts_model = None
    def train(self, X_arrays, y_pass, y_atts):
        S,X,U = X_arrays
        if self.pass_model is None:
            self.pass_model = build_pass_model(S.shape[1],X.shape[1], U.shape[1])
            self.atts_model = build_atts_model(S.shape[1], X.shape[1], U.shape[1])
        self.pass_model.fit([S,X,U], y_pass)
        self.atts_model.fit([S, X, U], y_atts)
    def predict(self, student_data):
        S,X,U = student_data
        pass_proba = self.pass_model.predict([S,X,U])
        atts_estim = self.atts_model.predict([S,X,U])
