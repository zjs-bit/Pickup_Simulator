
class Player:
    """
    Class carrying data about an individual player skill and experience

    """
    def __init__(self,name,games_played=0,base_rating=60.0,**kwargs) -> None:
        self.name = name
        self.games_played=games_played
        self.base_rating = base_rating
        self.current_rating = base_rating

        if 'true_rating' in kwargs:
            self.true_rating = kwargs['true_rating']

        if 'true_off' in kwargs:
            self._true_off = kwargs['true_off']

        if 'true_def' in kwargs:
            self.true_def = kwargs['true_def']

        if 'take_three_prob' in kwargs:
            self.three_prob = kwargs['three_prob']

        if 'true_two_point' in kwargs:
            self._true_two_pt = kwargs['true_two_pt']

        if 'true_three_point' in kwargs:
            self._true_three_pt = kwargs['true_three_pt']

    @property
    def true_off(self):
        return self._true_off
    
    @true_off.setter
    def true_off(self,y):
        self._true_off=y

    @property
    def true_three_pt(self):
        return self._true_three_pt
    
    @true_three_pt.setter
    def true_off(self,y):
        self._true_off = self._true_off + y - self._true_three_pt
        self._true_three_pt=y

    @property
    def true_two_pt(self):
        return self._true_two_pt
    
    @true_two_pt.setter
    def true_off(self,y):
        self._true_off = self._true_off + y - self._true_two_pt
        self._true_two_pt=y


        
