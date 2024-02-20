
import tensorflow as tf
import numpy as np

class RatingUpdater:
#Class for updating the ratings of pick-up game players after game results

    def __init__(self,base_learning_rate = 1.0, learn_decay = lambda: np.log + 1) -> None:
        
        self.base_lr = base_learning_rate
        self.learn_decay = learn_decay
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def log_MOV_update(self,t1_ratings,t1_gp,t2_ratings,t2_gp,game_outcome):

        gameto = np.max(game_outcome)
        t1marg = game_outcome[0]-game_outcome[1]+gameto
        team_size = len(t1_ratings)
        exp_marg = self._sigmoid((sum(t1_ratings)-sum(t2_ratings))/team_size)*gameto*2

        t1updates = (t1marg - exp_marg)/team_size * [self.base_lr * 1/self.learn_decay(n) for n in t1_gp]
        t2updates = (exp_marg - t1marg)/team_size * [self.base_lr * 1/self.learn_decay(n) for n in t2_gp]

        return {"t1updated": np.array(t1_ratings)+np.array(t1updates),
                "t2updated": np.array(t2_ratings)+np.array(t2updates)}

    def log_MOV_update_tf(self,t1_ratings,t1_gp,t2_ratings,t2_gp,game_outcome):

        gameto = np.max(game_outcome)
        t1marg = game_outcome[0]-game_outcome[1]+gameto
        team_size = len(t1_ratings)
        outfrac = t1marg/gameto*2

        t1tens = tf.Variable(t1_ratings)
        t2tens = tf.Variable(t2_ratings)

        with tf.GradientTape() as tape:
            linear_term  = (sum(t1tens) - sum(t2tens))/team_size
            loss = gameto*tf.nn.sigmoid_cross_entropy_with_logits(labels=[outfrac],logits=[linear_term])

        return tape.gradient(loss,[t1tens,t2tens])


r_updater = RatingUpdater()
