from typing import Any
from Agents.Random_Agent import RandomAgent
from Agents.MaxDamage_Agent import MaxDamageAgent
from poke_env import Player
from poke_env.battle import AbstractBattle
import numpy as np


class MixedAgent(Player):
    '''Mixed Agent switches between RandomAgent and Maxdamage Agent randomly each battle'''
    def __init__(self, p: float = .5, **kwargs: Any):
        super().__init__(**kwargs)

        self.p = p  #probability random is chosen
        self.RandomAgent = RandomAgent() 
        self.MaxDamageAgent = MaxDamageAgent() 
        self.RandomPlaying = True 
    
    async def _create_battle(self, split_message):
        '''Augmented create battle function with
        added functionality of choosing one of our two agents'''
        if np.random.rand() < self.p:
            self.RandomPlaying = True
        else:
            self.RandomPlaying = False
        return await super()._create_battle(split_message)

    def choose_move(self, battle):
        '''Chooses our move randomly'''
        if self.RandomPlaying: 
            return self.RandomAgent.choose_move(battle)
        else: 
            return self.MaxDamageAgent.choose_move(battle)


        