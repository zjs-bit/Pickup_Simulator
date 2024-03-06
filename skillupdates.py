
import tensorflow as tf
import numpy as np

class RatingUpdater:
#Class for updating the ratings of pick-up game players after game results

    def __init__(self,base_learning_rate = 1.0, learn_decay = lambda n: np.log(n) + 1,C=10.0) -> None:
        
        self.base_lr = base_learning_rate
        self.learn_decay = learn_decay
        self.C = C
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def log_MOV_update(self,t1_ratings,t1_gp,t2_ratings,t2_gp,game_outcome):

        gameto = np.max(game_outcome)
        t1marg = game_outcome[0]-game_outcome[1]+gameto
        team_size = len(t1_ratings)
        exp_marg = self._sigmoid((sum(t1_ratings)-sum(t2_ratings))/(team_size*self.C))*gameto*2

        t1updates = (t1marg - exp_marg)/(team_size*self.C) * np.array([self.base_lr * 1/self.learn_decay(n) for n in t1_gp])
        t2updates = (exp_marg - t1marg)/(team_size*self.C) *np.array([self.base_lr * 1/self.learn_decay(n) for n in t2_gp])

        print(t1updates)

        return {"t1updated": np.array(t1_ratings)+np.array(t1updates),
                "t2updated": np.array(t2_ratings)+np.array(t2updates)}

    def log_MOV_update_tf(self,t1_ratings,t1_gp,t2_ratings,t2_gp,game_outcome):

        gameto = max(game_outcome)
        t1marg = game_outcome[0]-game_outcome[1]+gameto
        team_size = float(len(t1_ratings))
        outfrac = t1marg/(gameto*2)

        t1tens = tf.Variable(t1_ratings)
        t2tens = tf.Variable(t2_ratings)

        with tf.GradientTape() as tape:
            linear_term  = (tf.reduce_sum(t1tens) - tf.reduce_sum(t2tens))/(team_size*self.C)
            print(linear_term, outfrac)
            loss = -(gameto*2)*tf.nn.sigmoid_cross_entropy_with_logits(labels=[outfrac],logits=[linear_term])

        return tape.gradient(loss,[t1tens,t2tens])
    
    def ctmc_grad(self,t1_ratings,t2_ratings,t1score,t2score,winbytwo=True,gameto=-1):

        if gameto ==-1:
            gameto = max(t1score,t2score)

        team_size=len(t1_ratings)
        t1_ratings=tf.Variable(t1_ratings)
        t2_ratings=tf.Variable(t2_ratings)

        with tf.GradientTape() as tape:
            z = (tf.reduce_sum(t1_ratings)-tf.reduce_sum(t2_ratings))/(team_size*self.C)
            t1_nextpt_prob = tf.math.sigmoid(z)
        
        mc_results  = self._mc_winprob(t1_nextpt_prob.numpy(),t1score,t2score,winbytwo,gameto=gameto)
        print (mc_results['grads'][(0,0)])

        print (tape.gradient(t1_nextpt_prob,t1_ratings)*mc_results['grads'][(0,0)])
        print (tape.gradient(t1_nextpt_prob,t2_ratings)*mc_results['grads'][(0,0)])


    def _mc_winprob(self,prob_t1,t1score,t2score,winbytwo,**kwargs):

        result_dict = {(t1score,t2score):1}
        derivative_dict  = {(t1score,t2score):0}
        t1win = (t1score > t2score)*1

        for i in range(t1score+1*(1-t1win)):
            for j in range(t2score+1*(t1win)):

                eval_i=t1score-(i+t1win)
                eval_j = t2score-(j+1-t1win)

                if (eval_i, eval_j) in result_dict:
                    continue

                if winbytwo and 'gameto' in kwargs:
                    if max(eval_i,eval_j)>kwargs['gameto'] and abs(eval_i-eval_j)>1:
                        result_dict[(eval_i, eval_j)] = 0
                        continue

                if (eval_i+1, eval_j) in  result_dict:
                    score1_next = result_dict[(eval_i+1, eval_j)]
                    deriv1_next = derivative_dict[(eval_i+1, eval_j)]
                else:
                    score1_next = 0
                    deriv1_next =0

                if (eval_i,  eval_j+1) in  result_dict.keys():
                    score2_next = result_dict[(eval_i, eval_j+1)]
                    deriv2_next = derivative_dict[(eval_i, eval_j+1)]
                else:
                    score2_next = 0
                    deriv2_next = 0 

                result_dict[(eval_i, eval_j)]=prob_t1*score1_next+(1-prob_t1)*score2_next
                derivative_dict[(eval_i, eval_j)] = score1_next - score2_next + prob_t1*deriv1_next + (1-prob_t1)*deriv2_next

        return {"probs":result_dict, "grads":derivative_dict}


r_updater = RatingUpdater()

#example 1 on 1 game
t1skill = [100.0,70.0]
t2skill = [50.0,80.0]

t1games = [1,1]
t2games = [1,1]

outcome = [10,7]


#r_updater.log_MOV_update(t1skill,t1games,t2skill,t2games,outcome)
#print(r_updater.log_MOV_update_tf(t1skill,t1games,t2skill,t2games,outcome))

p1  = .55

#print(r_updater._ctmc_winprob(p1,10,7))
r_updater.ctmc_grad([4.0],[2.0],10,7)

