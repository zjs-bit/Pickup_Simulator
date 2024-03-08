class Game:

    """
    Class representing a game of basketball between two teams and associated data.
    
    Inputs in construction include the players on each team (class Player), 
    the target score to win, the win condition (e.g., win by 2), the type of scoring  (e.g., 1s and 2s),
    and possesion rules (e.g., make-it-take-it)

    """
    def __init__(self,t_one,t_two,
                 target_score = 11, 
                 win_by_two  = True, 
                 score_inc = 'ones & twos',
                 makeit_takeit = False) -> None:
        
        self.t_one = t_one
        self.t_two = t_two
        self.target_score = target_score
        self.win_by_two = win_by_two
        self.makeit_takeit = makeit_takeit

        if score_inc not in ('ones','ones & twos','two & threes'):
            raise ValueError("score_inc must be in ('ones','ones & twos','two & threes')")
        else: 
            self.score_inc = score_inc