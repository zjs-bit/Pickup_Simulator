
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


r_updater = RatingUpdater()

#example 1 on 1 game
t1skill = [100.0,70.0]
t2skill = [50.0,80.0]

t1games = [1,1]
t2games = [1,1]

outcome = [10,7]


r_updater.log_MOV_update(t1skill,t1games,t2skill,t2games,outcome)
print(r_updater.log_MOV_update_tf(t1skill,t1games,t2skill,t2games,outcome))
