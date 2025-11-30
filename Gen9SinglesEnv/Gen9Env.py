from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv, SingleAgentWrapper, ActionType, ObsType
from typing import Tuple, Any, Dict, Awaitable
from poke_env.battle.status import Status
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType 

import numpy as np

from gymnasium.spaces import Box, Discrete


type_chart = {
    'NORMAL': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 0.5, 'GHOST': 0, 'DRAGON': 1, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'FIRE': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 0.5, 'ELECTRIC': 1, 'GRASS': 2, 'ICE': 2, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 2, 'ROCK': 0.5, 'GHOST': 1, 'DRAGON': 0.5, 'DARK': 1, 'STEEL': 2, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'WATER': {'NORMAL': 1, 'FIRE': 2, 'WATER': 0.5, 'ELECTRIC': 1, 'GRASS': 0.5, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 2, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 2, 'GHOST': 1, 'DRAGON': 0.5, 'DARK': 1, 'STEEL': 1, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'ELECTRIC': {'NORMAL': 1, 'FIRE': 1, 'WATER': 2, 'ELECTRIC': 0.5, 'GRASS': 0.5, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 0, 'FLYING': 2, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 0.5, 'DARK': 1, 'STEEL': 1, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'GRASS': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 2, 'ELECTRIC': 1, 'GRASS': 0.5, 'ICE': 1, 'FIGHTING': 1, 'POISON': 0.5, 'GROUND': 2, 'FLYING': 0.5, 'PSYCHIC': 1, 'BUG': 0.5, 'ROCK': 2, 'GHOST': 1, 'DRAGON': 0.5, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'ICE': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 0.5, 'ELECTRIC': 1, 'GRASS': 2, 'ICE': 0.5, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 2, 'FLYING': 2, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 2, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'FIGHTING': {'NORMAL': 2, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 2, 'FIGHTING': 1, 'POISON': 0.5, 'GROUND': 1, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'BUG': 0.5, 'ROCK': 2, 'GHOST': 0, 'DRAGON': 1, 'DARK': 2, 'STEEL': 2, 'FAIRY': 0.5, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'POISON': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 2, 'ICE': 1, 'FIGHTING': 1, 'POISON': 0.5, 'GROUND': 0.5, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 0.5, 'GHOST': 0.5, 'DRAGON': 1, 'DARK': 1, 'STEEL': 0, 'FAIRY': 2, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'GROUND': {'NORMAL': 1, 'FIRE': 2, 'WATER': 1, 'ELECTRIC': 2, 'GRASS': 0.5, 'ICE': 1, 'FIGHTING': 1, 'POISON': 2, 'GROUND': 1, 'FLYING': 0, 'PSYCHIC': 1, 'BUG': 0.5, 'ROCK': 2, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'STEEL': 2, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'FLYING': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 0.5, 'GRASS': 2, 'ICE': 1, 'FIGHTING': 2, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 2, 'ROCK': 0.5, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'PSYCHIC': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 2, 'POISON': 2, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 0.5, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 1, 'DARK': 0, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'BUG': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 2, 'ICE': 1, 'FIGHTING': 0.5, 'POISON': 0.5, 'GROUND': 1, 'FLYING': 0.5, 'PSYCHIC': 2, 'BUG': 1, 'ROCK': 1, 'GHOST': 0.5, 'DRAGON': 1, 'DARK': 2, 'STEEL': 0.5, 'FAIRY': 0.5, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'ROCK': {'NORMAL': 1, 'FIRE': 2, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 2, 'FIGHTING': 0.5, 'POISON': 1, 'GROUND': 0.5, 'FLYING': 2, 'PSYCHIC': 1, 'BUG': 2, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'GHOST': {'NORMAL': 0, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 2, 'BUG': 1, 'ROCK': 1, 'GHOST': 2, 'DRAGON': 1, 'DARK': 0.5, 'STEEL': 1, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'DRAGON': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 2, 'DARK': 1, 'STEEL': 0.5, 'FAIRY': 0, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'DARK': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 0.5, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 2, 'BUG': 1, 'ROCK': 1, 'GHOST': 2, 'DRAGON': 1, 'DARK': 0.5, 'STEEL': 1, 'FAIRY': 0.5, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'STEEL': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 0.5, 'ELECTRIC': 0.5, 'GRASS': 1, 'ICE': 2, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 2, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'FAIRY': 2, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1, 'STEEL': 0.5},
    'FAIRY': {'NORMAL': 1, 'FIRE': 0.5, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 2, 'POISON': 0.5, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 2, 'DARK': 2, 'STEEL': 0.5, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'STELLAR': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'STEEL': 1, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
    'THREE_QUESTION_MARKS': {'NORMAL': 1, 'FIRE': 1, 'WATER': 1, 'ELECTRIC': 1, 'GRASS': 1, 'ICE': 1, 'FIGHTING': 1, 'POISON': 1, 'GROUND': 1, 'FLYING': 1, 'PSYCHIC': 1, 'BUG': 1, 'ROCK': 1, 'GHOST': 1, 'DRAGON': 1, 'DARK': 1, 'STEEL': 1, 'FAIRY': 1, 'STELLAR': 1, 'THREE_QUESTION_MARKS': 1},
}


class Gen9SinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define the observation space
        # We let our observation space be a 14-dimensional vector
        # The first four entries are the moves' base powers (or -1 if no move)
        # The next four are the damage multipliers for each move we have
        # The ninth entry is the number of remaining Pokemon
        # The tenth entry is the number of opponent's remaining Pokemon
        # The eleventh entry is the number of pokemon statused on our side
        # The twielfth entry is the number of pokemon statused on opponent's side
        # The thirteenth and fourteenth entries are for fraction hp of our active pokemon and opponent's active pokemon
        #The fifteenth entry is speed control (0 if slower than opponent, 1 if faster)
        #The sixteenth to nineteenth entries are priority values for each move
        #The twentienth to twentythird entries are move categories for each move (0=physical, 1=special, -1=status)
        #The first six entries are the typing for all of our available pokemon
        #The Last entries are all for the opponent active pokemon typing (0 if typing is false 1 otherwise)
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, -1, -1, -1, -1, -1,
                -1, -1, -1] + 6 * (18 * [0]) + 18 * [0]
        high = [3, 3, 3, 3, 4, 4, 4, 4,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1] + 6 * (18 * [1]) + 18 * [1]

        #Experimenting by removing the action mask from observation space

        # Define the observation space
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        # our observation space is a continuous Box space with 14 dimensions
        # Meaning each entry is a float32, it's essentially a vector of 14 float32 numbers
        # Each 'dimension' corresponds to a specific feature of the battle state
        # The Box space allows us to represent a range of values for each feature

        self.type_chart = type_chart

        self.n_actions = 26 # 6 switches + 4 moves + 4 mega + 4 z + 4 dynamax + 4 tera
        #Note that mega, z, and dynamax actions will never be taken

        self.special_moves = {"struggle", "recharge"}

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Returns reward for current Battle State"""
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=10.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Embeds our battle into an observation space"""
        moves_base_power, damage_multipliers = self._Embed_base_power_and_multiplier(battle)

        #Vestial remaining pokemon
        our_remaining_pokemon, opponent_remaining_pokemon, our_statused_pokemon, opponent_statused_pokemon = self._Embed_status_and_remaining_pokemon(battle)

        #Normalize statuses to 0-1
        our_statused_pokemon = our_statused_pokemon / 6
        opponent_statused_pokemon = opponent_statused_pokemon /6

        #Get fraction hp of our active pokemon (normalized 0–1)
        our_active_fraction_hp = battle.active_pokemon.current_hp_fraction

        #Get fraction hp of opponent's active pokemon (normalized 0–1)
        opponent_active_fraction_hp = battle.opponent_active_pokemon.current_hp_fraction

        #Encoding speed control (1 if we are faster, 0 if slower)
        speed_control = self._Embed_speed_control(battle)
        
        #Encode priority of each move
        priority_values, move_categories = self._Embed_move_priority_and_category(battle)

        #Embed the current action mask into the observation
        #action_mask = self.get_action_mask(battle)
    
        # Concatenate all features into a single observation vector
        observation = np.concatenate(
            [
                moves_base_power,
                damage_multipliers,
                [our_statused_pokemon],
                [opponent_statused_pokemon],
                [our_active_fraction_hp],
                [opponent_active_fraction_hp],
                [speed_control],
                priority_values,
                move_categories,
            ]
        )

        #Concatenate the typing observation for each of our pokemon
        for mon in battle.team.values():
            observation = np.concatenate([observation, self._embed_typing(mon)])

        #Concatenate the typing oservation for opponents active pokemon
        observation = np.concatenate([observation, self._embed_typing(battle.opponent_active_pokemon)])

        return observation.astype(np.float32)
    
    def _Embed_base_power_and_multiplier(self, battle: AbstractBattle) -> Tuple[np.ndarray, np.ndarray]:
        '''Helper function to embed base power and damage multipliers for available moves'''
        moves_base_power = -np.ones(4)      # base power values (normalized)
        damage_multipliers = np.ones(4)     # type effectiveness multipliers

        # Enumerate through available moves and populate the vectors
        for i, move in enumerate(battle.available_moves):
            # Normalize base power to be between 0 and 3 and update values. -1 if no move or move has no base power
            moves_base_power[i] = move.base_power / 100 if move.base_power is not None else -1
            # Update damage multiplier values for each move
            damage_multipliers[i] = move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
                type_chart=self.type_chart,
            )
        
        return moves_base_power, damage_multipliers
    
    def _Embed_status_and_remaining_pokemon(self, battle: AbstractBattle) -> Tuple[float, float, float, float]:
        '''Helper function to embed status and remaining pokemon counts'''
        num_remaining_pokemon = 0
        num_statused_pokemon = 0
        num_opponent_remaining_pokemon = 0
        num_opponent_statused_pokemon = 0

        for mon in battle.team.values():
            if not mon.fainted:
                num_remaining_pokemon += 1
                if mon.status is not None:
                    num_statused_pokemon += 1
        
        for mon in battle.opponent_team.values():
            if not mon.fainted:
                num_opponent_remaining_pokemon += 1
                if mon.status is not None:
                    num_opponent_statused_pokemon += 1
        
        return (num_remaining_pokemon, num_opponent_remaining_pokemon,
                num_statused_pokemon, num_opponent_statused_pokemon)

    def _Embed_speed_control(self, battle: AbstractBattle) -> float:
        '''Helper function to embed speed control feature'''
        our_speed = battle.active_pokemon.base_stats["spe"] * (1 + battle.active_pokemon.boosts["spe"] * .5)
        opponent_speed = battle.opponent_active_pokemon.base_stats["spe"] * (1 + battle.opponent_active_pokemon.boosts["spe"] * .5)

        if battle.active_pokemon.status == Status.PAR:
            our_speed = our_speed * .5
        if battle.opponent_active_pokemon.status == Status.PAR:
            opponent_speed = opponent_speed * .5

        return 1 if our_speed > opponent_speed else 0

    def _Embed_move_priority_and_category(self, battle: AbstractBattle) -> Tuple[np.ndarray, np.ndarray]:
        '''Helper function to embed move priority and category features'''
        priority_values = np.zeros(4)
        move_categories = np.zeros(4) # 1=physical, .5=special, -1=status, 0=no move

        for i, move in enumerate(battle.available_moves):
            if move._id in self.special_moves:
                priority_values[i] = 0  # Assign neutral priority for special moves
            else:
                priority_values[i] = move.priority / 7  # Normalize priority to be between -1 and 1

            if move.category == MoveCategory.PHYSICAL:
                move_categories[i] = 1
            elif move.category == MoveCategory.SPECIAL:
                move_categories[i] = .5
            else:
                move_categories[i] = -1

        return priority_values, move_categories
    
    def _embed_typing(self, mon) -> np.ndarray:
        '''embeds the active pokemon's typing for opponent'''
        typing_info = np.zeros(18)
        if mon.status == Status.FNT: 
            return typing_info
        types = mon.types
        if PokemonType.BUG in types:      typing_info[0]  = 1
        if PokemonType.DARK in types:     typing_info[1]  = 1
        if PokemonType.DRAGON in types:   typing_info[2]  = 1
        if PokemonType.ELECTRIC in types: typing_info[3]  = 1
        if PokemonType.FAIRY in types:    typing_info[4]  = 1
        if PokemonType.FIGHTING in types: typing_info[5]  = 1
        if PokemonType.FIRE in types:     typing_info[6]  = 1
        if PokemonType.FLYING in types:   typing_info[7]  = 1
        if PokemonType.GHOST in types:    typing_info[8]  = 1
        if PokemonType.GRASS in types:    typing_info[9]  = 1
        if PokemonType.GROUND in types:   typing_info[10] = 1
        if PokemonType.ICE in types:      typing_info[11] = 1
        if PokemonType.NORMAL in types:   typing_info[12] = 1
        if PokemonType.POISON in types:   typing_info[13] = 1
        if PokemonType.PSYCHIC in types:  typing_info[14] = 1
        if PokemonType.ROCK in types:     typing_info[15] = 1
        if PokemonType.STEEL in types:    typing_info[16] = 1
        if PokemonType.WATER in types:    typing_info[17] = 1

        return typing_info



    def get_action_mask(self, battle: AbstractBattle):
        '''returns a vector of valid actions for the current battle state'''
        action_mask = np.zeros(26, dtype=np.float32) # initialize all actions as invalid

        #Map each pokemon to an index 0-5
        index_to_pokemon = {i: mon for i, mon in enumerate(battle.team.values())}


        #First mark available switches as valid 0-5
        if not battle.trapped:
            for i, mon in index_to_pokemon.items():
                if mon in battle.available_switches:
                    action_mask[i] = 1

        #Then mark the valid moves 6-9
        if not battle.force_switch:
            if battle.active_pokemon is not None:
                #Map each move to an index 6-9
                index_to_move = {i + 6: move for i, move in enumerate(battle.active_pokemon.moves.values())}
                for i, move in index_to_move.items():
                    if move in battle.available_moves:
                        action_mask[i] = 1

                        '''moves 10-13 are mega evolutions
                        if battle.can_mega_evolve:
                            valid_actions[i + 4] = 1
                        
                        #moves 14-17 are z moves
                        if battle.can_z_move:
                            valid_actions[i + 8] = 1

                        #moves 18-21 are dynamax
                        if battle.can_dynamax:
                            valid_actions[i + 12] = 1
                        '''

                        #I'm leaving mega evoltions, z moves, and dynamax out for now
                        #The reason being the format for this current but doesn't require them
                        #And I am concerned that Poke-env will read them as possible with this implementation
                        #In the even that they are needed later, we can add them back in
                        
                        #moves 22-25 are tera
                        if battle.can_tera:
                            action_mask[i + 16] = 1
        return action_mask
    

class MaskedSingleAgentWrapper(SingleAgentWrapper):
    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        assert self.env.battle1 is not None
        assert self.env.battle2 is not None
        opp_order = self.opponent.choose_move(self.env.battle2)
        assert not isinstance(opp_order, Awaitable)
        opp_action = self.env.order_to_action(
            opp_order, self.env.battle2, fake=self.env.fake, strict=self.env.strict
        )
        #Incorporate action masking into our step function
        action_mask = self.env.get_action_mask(self.env.battle1)
        action = int(action)
        is_invalid = (action_mask[action] == 0)
        if is_invalid:
            # If the chosen action is invalid, choose a random valid action
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            if valid_actions:
                action = int(np.random.choice(valid_actions))
            else:
                #If no valid actions (should not happen), choose random action
                action = int(self.action_space.sample())

        action = np.int64(action)
        #Proceed with the original step function
        actions = {
            self.env.agent1.username: action,
            self.env.agent2.username: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        if is_invalid: rewards[self.env.agent1.username] = rewards[self.env.agent1.username] - .5 #slight negative penalty for picking invalid option.
        #We will eventually remove this for true action masking.
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )