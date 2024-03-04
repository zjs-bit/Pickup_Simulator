
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
    
    def ctmc_grad(self,t1_off,t1_def,t2_off,t2_def,game_outcome):

        t1_rate = np.exp(sum(t1_off)-sum(t2_def))
        t2_rate = np.exp(sum(t2_off)-sum(t1_def))
        prob_t1= t1_rate/(t1_rate+t2_rate)

        prob_t1 =  tf.Variable([.55])

        with tf.GradientTape() as tape:
            p_win_t1 = self._ctmc_winprob(prob_t1,game_outcome)[(0,0)]

            print(tape.gradient(p_win_t1,[prob_t1]))

    def _ctmc_winprob(self,prob_t1,game_outcome):

        result_dict = {tuple(game_outcome):1}

        for i in range(game_outcome[0]+1):
            for j in range(game_outcome[1]+1):

                eval_i=game_outcome[0]-i
                eval_j = game_outcome[1]-j

                if (eval_i,eval_j) in result_dict.keys():
                    continue

                if (eval_i+1,eval_j) in  result_dict.keys():
                    score1_next = result_dict[(eval_i+1,eval_j)]
                else:
                    score1_next = 0

                if (eval_i,eval_j+1) in  result_dict.keys():
                    score2_next = result_dict[(eval_i,eval_j+1)]
                else:
                    score2_next = 0

                result_dict[(eval_i,eval_j)]=prob_t1*score1_next+(1-prob_t1)*score2_next

        return result_dict


        

    


r_updater = RatingUpdater()

#example 1 on 1 game
t1skill = [100.0,70.0]
t2skill = [50.0,80.0]

t1games = [1,1]
t2games = [1,1]

outcome = [10,7]


#r_updater.log_MOV_update(t1skill,t1games,t2skill,t2games,outcome)
#print(r_updater.log_MOV_update_tf(t1skill,t1games,t2skill,t2games,outcome))

p1  = .9

print(r_updater._ctmc_winprob(p1,outcome))
r_updater.ctmc_grad([4],[2],[3],[5],outcome)

