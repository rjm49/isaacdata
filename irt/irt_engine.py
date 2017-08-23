'''
Created on 3 May 2017

@author: Russell
'''
import math

class IRTEngine(object):

    def __init__(self):
        self.history = []
        self.curr_theta = 3
        self.MAX_THETA = 6
        self.MIN_THETA = 0

    def prob_of_correct_ans(self, theta, beta):
        prob = 1.0 / (1.0+ math.exp(beta - theta))
        if prob<0 or prob>1:
            exit()
        return prob

    def calc_better_theta(self, curr_theta, qs_used):
        eps = 0.1
        sum_spread=0.0
        sum_info = 0.0
        for q_i in qs_used:
            #go through the answered questions, sum up the difference in value vs expected value
            u_i = float( q_i["passed"] ) #cast bool of response to float i.e. {1,0}
            p_i_theta = self.prob_of_correct_ans(curr_theta, q_i["beta"])
    #         print("real/exp",u_i,p_i_theta)
            sum_spread += ( u_i - p_i_theta ) #TODO check this -ve sign
            #Then get the information function, tells us about precision of estimate
            info = p_i_theta * (1.0-p_i_theta)
            sum_info += info
    #     print("sum_spread=",sum_spread, "scaled_sp=",(sum_spread/sum_info))
    #     print("total info", sum_info)
        new_theta = curr_theta + (sum_spread / sum_info)
        print("raw new theta=", new_theta)
        
        if new_theta >=self.MAX_THETA:
            new_theta = self.MAX_THETA
            return new_theta
        elif new_theta <=self.MIN_THETA:
            new_theta = self.MIN_THETA
            return new_theta
        elif abs(new_theta - curr_theta) < eps:
            return new_theta # we're done, new theta is calc'd
        else:
            return self.calc_better_theta(new_theta, qs_used) # iterate again

        
    def update(self, level, passed):
        qobj = {}
        qobj["beta"]=level
        qobj["passed"]=passed
        self.history.append(qobj)
        self.current_theta = self.calc_better_theta(self.curr_theta, self.history)
        return self.current_theta