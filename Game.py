from dataclasses import dataclass, field
from Player import Player
from typing import List
import numpy as np
@dataclass
class Game:
    """
    Class representing a game of basketball between two teams and associated data.
    
    Inputs in construction include the players on each team (class Player), 
    the target score to win, the win condition (e.g., win by 2), the type of scoring  (e.g., 1s and 2s),
    and possesion rules (e.g., make-it-take-it)

    """
    t_one: List[Player]
    t_two: List[Player]
    target_score : int = 11
    win_by_two : bool = True
    makeit_takeit : bool = False
    max_score_increment: int = 2
    team_size: int = field(init=False)
    t1_ratings: np.array = field(init = False)
    t2_ratings: np.array = field(init = False)
    t1_sd: np.array = field(init = False)
    t2_sd: np.array = field(init = False)
    player_ratings: dict = field(init = False)

    def __post_init__(self):
        self.team_size = len(self.t_one)
        self.t1_ratings = np.array([p.current_rating for p in self.t_one])
        self.t2_ratings = np.array([p.current_rating for p in self.t_two])
        self.t1_sd = np.array([p.current_sd for p in self.t_one])
        self.t2_sd= np.array([p.current_sd for p in self.t_two])
        self.player_ratings = {"t1_ratings": self.t1_ratings,
                "t2_ratings": self.t2_ratings}

    def add_result(self,t1score: int,t2score: int):
        self.t1_score = t1score
        self.t2_score = t2score
        self.t1_margin = t1score - t2score

    

if __name__ == '__main__':   
    player1 = Player("Joe")
    player2 = Player("Cap")
    g = Game([player1],[player2])
    print(g.player_ratings)

    c = [1,2]

    b = [2,4]

    print([z+x for z,x in zip(c,b)])
       

