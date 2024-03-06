
class Player:
    """
    Class carrying data about an individual player skill and experience

    """
    def __init__(self,name,games_played=0,base_rating=60.0) -> None:
        self.name = name
        self.games_played=games_played
        self.base_rating = base_rating
        self.current_rating = base_rating 

