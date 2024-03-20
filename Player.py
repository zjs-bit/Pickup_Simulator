import numpy as np
class Player:
    """
    Class carrying data about an individual player skill and experience

    """
    #Sampling information for random player creation
    off_weight = .7
    def_weight = .3
    take_3_alpha = 4.0
    take_3_beta = 12.0

    sd_def = 4
    sd_2pt = 5
    sd_3pt = 6
    sd_off_intercept = 5
    
    corr_2pt_3pt = .5
    cov_2pt_3pt = corr_2pt_3pt*sd_2pt*sd_3pt
    off_cov = np.array([[sd_2pt**2,cov_2pt_3pt],
                        [cov_2pt_3pt,sd_3pt**2]])
    def_shift_big = 5

    def __init__(self,name,games_played=0,base_rating:float = 60.0,base_sd:float = 10.0,**kwargs) -> None:
        self.name = name
        self.games_played=games_played
        self.base_rating = base_rating
        self.base_sd  = base_sd

        #These could be vectors if more than one player characteristic is tracked by Elo (e.g., offense and defense)
        self.current_rating = base_rating
        self.current_sd = base_sd
        

        if 'height' in kwargs:
            self.height  = kwargs['height']
        
        if 'position' in kwargs:
            self.position = kwargs['position']

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
    def true_three_pt(self,y):
        self._true_off = self._true_off + y - self._true_three_pt
        self._true_three_pt=y

    @property
    def true_two_pt(self):
        return self._true_two_pt
    
    @true_two_pt.setter
    def true_two_pt(self,y):
        self._true_off = self._true_off + y - self._true_two_pt
        self._true_two_pt=y

    def assign_random_attributes(self):

        shooting_sample = np.random.multivariate_normal(np.array([0,0]),Player.off_cov)
        self._true_three_pt = shooting_sample[1]
        self._true_two_pt = shooting_sample[0]
        off_intercept = np.random.normal(0,Player.sd_off_intercept)
        self._true_off = self._true_three_pt+self._true_three_pt+off_intercept
        self.true_def = np.random.normal(0,Player.sd_def)
        self.three_prob = np.random.beta(Player.take_3_alpha,Player.take_3_beta)
        self.true_rating = Player.def_weight*self.true_def + Player.off_weight*self.true_off




