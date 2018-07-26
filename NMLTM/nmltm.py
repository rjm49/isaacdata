import keras
from keras import Model, Input
from keras.layers import Dense, multiply, add
from keras.optimizers import Adam


def init_nmltm_model(w_evidence=11000, w_predictions=11000, n_traits=300):
    prior_exp = Input(1)
    student_skill = Input(1)
    practice = Input(w_evidence)
    skill_diff = Dense(n_traits)
    learning_rates = Dense(n_traits)

    gT = multiply([learning_rates, practice])
    b_gT = add([skill_diff, gT])
    th_b_gt = add([student_skill, b_gT], activation="signmoid")

    outp = Dense(1)(th_b_gt, activation="sigmoid")


    o = Adam()
    m = Model(inputs=inp, outputs=outp)
    m.compile(optimizer=o, loss='binary_crossentropy')

class NMLTM():
    def __init__(self, w_e, sxua, scaler, rd):
        self.model = init_nmltm_model(w_e=w_e, w_o=1, n_traits=300)
        self.sxua = sxua
        self.scaler = scaler
        self.reverse_dict = rd