
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.stats import binom
from Game import Game
from Player import Player
import pandas as pd
from functools import lru_cache


class EloModel:
    def __init__(self,
                 learning_rate:float = 1.0,
                 C:float=10.0,
                 score_buffer: int = 5,
                 mode = "deterministic") -> None:
        
        self.learning_rate  = learning_rate
        self.C = C
        self.score_buffer = score_buffer
        if mode not in ('deterministic','bayesian'):
            raise ValueError('mode not in ("deterministic","bayesian")')
        self.mode = mode

    def learn_decay(self,games_played: int) -> float:
        return np.log(games_played+1) + 1
    
    def reachable_states(self,game:Game)-> tuple:
        """Returns the reachable states in a game. Defined in subclass"""
        pass

    def is_terminal_state(self,reachable_state)->bool:
        """Returns true if reachable_state is a terminal (game completed) state. Defined in subclass"""
        pass

    def _get_MOV(self,terminal_state) -> int:
        pass

    def _get_t1Win(self,terminal_state)-> bool:
        pass

    def likelihood(self,reachable_state,game:Game) -> float:
        """Method returning the likelihood of observing the outcome of game  
        game: Holds relevant data on the game;
        outcome: a dictionary storing relevant information for a possible outcome of the game 
        required for computing the likelihood
        """

    def pregame_outcome_probs(self,game: Game) -> dict:
        """Computes a dictionary containing the probability of potential outcomes for the 
        matchup in "game" using self.likelihood()
        """
        outcomes = self.reachable_states(game)
        return self._pregame_outcome_probs_cached(outcomes)
        
    @lru_cache(maxsize=5)
    def _pregame_outcome_probs_cached(self,outcomes)->dict:
        return {outcome:self.likelihood(outcome,game) for outcome in outcomes}

    def pregame_MOV_probs(self, game: Game)->pd.DataFrame:
        outcome_probs = self.pregame_outcome_probs(game)
        outprobs = {k:outcome_probs[k] for k in outcome_probs.keys() if self.is_terminal_state(k)}
        print(outprobs)
        predict_mov = pd.DataFrame.from_dict({'outcomes':outprobs.keys(),'prob':outprobs.values()})
        predict_mov['MOV'] = predict_mov['outcomes'].apply(self._get_MOV)
        return predict_mov.groupby('MOV').agg(MOV_prob=('prob',np.sum))

    def pregame_win_prob(self, game: Game) -> float:
        """Returns the pregame probability that team 1 will win"""
        outprobs = dict(filter(self.pregame_outcome_probs(game),self.is_terminal_state))
        return sum(dict(filter(outprobs,self._get_t1Win)).values())

    def pregame_predicted_MOV(self, game: Game) -> pd.DataFrame:
       MOV_tab =  self.pregame_MOV_probs(game)
       return pd.sum(MOV_tab['MOV']*MOV_tab['prob'])

    def postgame_likelihood(self, game: Game):
        """Returns the likelihood of the observed outcome in game, if available

        game: Game object storing the players and setting for the game
        game.result: stores a generic results dictionary storing the applicable outcomes observed
        at the end of the game
        """

        if hasattr(game, 'result'):
            return self.likelihood(game=game,**game.result)
        else: 
            raise AttributeError("game object does not have an observed result assigned. \
                                 Use game.add_result to assign an outcome")

    def loss_function(self,game: Game):
        return -tf.math.log(self.postgame_likelihood(game))
    
    def gradient_step(self,game: Game) -> list:
        t1x = tf.Variable(game.t1_ratings)
        t2x = tf.Variable(game.t2_ratings)
        with tf.GradientTape() as tape:
            tape.watch([t1x,t2x])
            loss = self.loss_function(t1x,t2x,game)
        return [tm_grad.numpy() for tm_grad in tape.gradient(loss,[t1x,t2x])]
    
    def _normal_priors(self,game: Game) -> tfp.distributions.Normal:
        t1prior = tfd.Normal(loc=t1_m, scale=t1_var)
        t2prior = tfd.Normal(loc=t2_m, scale=t2_var)

        return t1prior, t2prior
    
    def _bayesian_update(self,t1x,t1var,t2x,t2var,optimizer: str,game: Game):
        t1priors,t2priors = self._normal_priors(t1x,t1var,t2x,t2var)

        @tf.function
        def objective(tx1,tx2):
            neglogprior = - (tf.reduce_sum(tf.math.log(t1priors.prob(t1x)))+tf.reduce_sum(tf.math.log(t2priors.prob(t2x))))
            loss = self.loss_function(t1x,t2x) + neglogprior
            return loss

        @tf.function
        def grad(tx1,tx2):
            with tf.GradientTape() as tape1:
                tape1.watch(tx1,tx2)
                loss = objective(tx1,tx2)
            return tape1.gradient(loss,tx1,tx2)
        
        @tf.function
        def hess(tx1,tx2):
            with tf.GradientTape() as tape2: 
                tape2.watch(tx1,tx2)
                g=grad(tx1,tx2)
            h = tape2.jacobian(g,[t1x,t2x])

            return h

    def skill_updates_det(self,t1x,t2x,game: Game):
        
        print(self.gradient_step(t1x,t2x,game))
        grads = self.gradient_step(t1x,t2x,game)
        expt1  = [p.games_played for p in game.t_one]
        expt2  = [p.games_played for p in game.t_two]
        t1updates = [c-z * self.learning_rate/self.learn_decay(gp) for c, z,gp in zip(game.t1_ratings, grads[0],expt1)]
        t2updates = [c-z * self.learning_rate/self.learn_decay(gp) for c, z,gp in zip(game.t2_ratings, grads[1],expt2)]

        return t1updates, t2updates
    
class OutcomeEloModel(EloModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def final_scores(self,game: Game) -> set:

        def reachable(t1scr,t2scr):
               (max(t1scr,t2scr)>game.target_score and (not game.win_by_two or game.win_by_two & abs(t1scr-t2scr)<=2))

        if game.win_by_two:
            max_score = game.target_score + self.score_buffer
        else: 
            max_score = game.target_score + game.max_score_increment - 1 

        return set([(i,j) for i in range(max_score) for j in range(max_score) if reachable(i,j)])
    
    def pregame_outcome_probs(self,game: Game) -> dict:
        """Computes a dictionary containing the probability of potential outcomes for the 
        matchup in "game" using self.likelihood()
        """
        outcomes = self.reachable_states(game)
        return {outcome:self.likelihood(*outcome,game) for outcome in outcomes}

    def pregame_MOV_probs(self, game: Game):
        outcome_probs = self.pregame_outcome_probs(game)
        predict_mov = pd.DataFrame(outcomes = outcome_probs.keys(), probs = outcome_probs.values())
        predict_mov['MOV'] = predict_mov['outcomes'].apply(lambda scr1, scr2: scr1 - scr2)
        return predict_mov.groupby('MOV').agg(MOV_prob=('probs',np.mean))

    def pregame_win_prob(self, game: Game):
        """Returns the pregame probability that team 1 will win"""
        outcome_probs = self.pregame_outcome_probs(game)
        winning_outcomes = dict(filter(lambda res: res[0] - res[1] > 0),outcome_probs.keys())
        return sum(winning_outcomes.values)


class MOVEloModel(EloModel):
    def __init__(self, cap_MOV_at_target_scr = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cap_MOV = cap_MOV_at_target_scr

    def reachable_states(self,game: Game) -> list: 
        if self.cap_MOV:
            return (i - game.target_score for i in range(2*game.target_score+1))
        else: 
            return ( - game.target_score for i in range(2*(game.target_score+game.max_score_increment)-1))
        
    def is_terminal_state(self, reachable_state) -> bool:
        return True
    
    def _get_MOV(self, terminal_state) -> int:
        return terminal_state
    
    def _get_t1Win(self, terminal_state) -> bool:
        return terminal_state > 0
    

class BinEloModel(MOVEloModel):
#Class for updating the ratings of pick-up game players after game results

    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.link = tf.math.sigmoid
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def lin_term(self,t1x,t2x):
        team_size = tf.shape(t1x).numpy()[0]
        return (tf.reduce_sum(t1x) - tf.reduce_sum(t2x))/(team_size*self.C)
    
    def bin_p(self,t1x,t2x):
        linterm = self.lin_term(t1x,t2x)
        return self.link(linterm)
    
    def likelihood(self, MOV: int, game: Game) -> float: #Restart here
        p = self.bin_p(game.t1_ratings,game.t2_ratings)
        MOV_trunc = min(abs(MOV),game.target_score)*np.sign(MOV)
        return binom.pmf(MOV_trunc + game.target_score,game.target_score*2,p)
    
    def _pregame_MOV_probs(self,t1x,t2x,game: Game):
        p=self.bin_p(t1x,t2x)
        t1_margin_probs = [binom.pmf(i,game.target_score*2,p) for i in range(game.target_score*2+1)]
        t1_margins = [i - game.target_score for i in range(game.target_score*2+1)]
        return {'Team1 margins': t1_margins,'Team1 margin probs': t1_margin_probs}

    def _pregame_win_prob(self,t1x,t2x,game:Game):
        p=self.bin_p(t1x,t2x)
        return 1-binom.cdf(game.target_score,game.target_score*2,p)

    def _pregame_pred_MOV(self,t1x,t2x,game :Game):
        p=self.bin_p(t1x,t2x)
        return p*game.target_score*2 - game.target_score

    def _postgame_likelihood(self,t1x,t2x,game):
        p=self.bin_p(t1x,t2x)
        adj_outcome = game.t1_margin+game.target_score
        bin_x = max(0,min(adj_outcome,game.target_score*2))
        return tfd.Binomial(total_count = game.target_score*2, probs = p).prob(bin_x)
    
    def _loss_function(self,t1x,t2x,game: Game):
        
        linear_term = self.lin_term(t1x,t2x)
        adj_outcome = max(0, min(game.t1_margin+game.target_score, game.target_score*2))
        outfrac = float(adj_outcome/(game.target_score*2))
        return (game.target_score*2)*tf.nn.sigmoid_cross_entropy_with_logits(labels=[outfrac],logits=[tf.cast(linear_term,dtype = tf.float32)])

    def _gradient_step(self,t1x: np.array,t2x: np.array ,game: Game):

        linear_denom = game.team_size*self.C
        adjoutcome = max(0, min(game.t1_margin+game.target_score, game.target_score*2))
        exp_marg = self._sigmoid((sum(t1x)-sum(t2x))/linear_denom)*game.target_score*2

        t1grads = -[(adjoutcome - exp_marg)/linear_denom for i in range(game.team_size)]
        t2grads = -[(exp_marg - adjoutcome)/linear_denom for i in range(game.team_size)]

        return t1grads, t2grads
    
class ctmc_Updater(EloModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
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
    
    def _mc_terminalprobs(self,prob_t1,winbytwo,gameto,max_score,**kwargs):

        result_dict = {(0,0):1}
        terminal_state = lambda i, j, win_by_two: max(i,j)>gameto and (not win_by_two or win_by_two & abs(i-j)>1)

        for i in range(gameto+self.max_margin+1):
            for j in range(i+1,gameto+self.max_margin+1-i):

                #Fill the row 
                if i-1 < 0 not in result_dict:
                    result_dict[(i,j)]=(1-prob_t1)*result_dict[(i,j-1)]*terminal_state(i,j-1)
                    result_dict[(j,i)]=prob_t1*result_dict[(j-1,i)]*terminal_state(j-1,i)
                else: 
                    result_dict[(i,j)]=prob_t1*result_dict[(i,j-1)]*terminal_state(i,j-1)+(1-prob_t1)*result_dict[(i-1,j)]*terminal_state(i-1,j)
                    result_dict[(j,i)]=prob_t1*result_dict[(j-1,i)]*terminal_state(j-1,i)+(1-prob_t1)*result_dict[(j,i-1)]*terminal_state(j,i-1)
        
        return result_dict


if __name__ == '__main__':

    p1 = Player("Joe",base_rating=75.0)
    p2 = Player("Bill",base_rating=60.0)

    p1.assign_random_attributes()
    p2.assign_random_attributes()

    game = Game([p1],[p2])

    game.add_result(11,8)

    zt=tf.Variable([75.0])
    z = tf.Variable([60.0])

    log_elo = BinEloModel()

    print(log_elo.reachable_states(game))
    z=log_elo.pregame_outcome_probs(game)

    print(log_elo.pregame_MOV_probs(game))
    














