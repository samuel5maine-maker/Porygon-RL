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

# Order defines the embedding index for each type
TYPE_ORDER = [
    PokemonType.BUG,
    PokemonType.DARK,
    PokemonType.DRAGON,
    PokemonType.ELECTRIC,
    PokemonType.FAIRY,
    PokemonType.FIGHTING,
    PokemonType.FIRE,
    PokemonType.FLYING,
    PokemonType.GHOST,
    PokemonType.GRASS,
    PokemonType.GROUND,
    PokemonType.ICE,
    PokemonType.NORMAL,
    PokemonType.POISON,
    PokemonType.PSYCHIC,
    PokemonType.ROCK,
    PokemonType.STEEL,
    PokemonType.WATER,
    PokemonType.STELLAR,
    PokemonType.THREE_QUESTION_MARKS
]

class Gen9SinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define the observation space
        # We let our observation space be a huge vector
        low = [-1, -1, -1, -1, #Move Base powers
               0, 0, 0, 0,  #Damage multipliers
               0, 0, 0, 0,  #stab flags
               0, 0, 0, 0,  #Healing flag
               0, 0, 0, 0,  #status move flag
               0, 0, 0, 0,  #is boost move
                0, 0,       #Our and opponent remaining pokemon
                0, 0,       #our and opponent statused pokemon
                0, 0,       #our and opponent fractional hp
                0,          #speed control (0 if false 1 if true)
                -1, -1, -1, -1, #priority values
                -1, -1, -1, -1, #move category 
                -1]              #Tera Type
        
        
        low = low + 14 * [-1] #oppont and our boosts
        
        low = low + 6 * (2 * [-1]) + 2 * [-1] #typing
        #6 pokemon two maximum types per pokemon plus typing for opponent pokemon

        high = [3, 3, 3, 3, #Move base powers
                4, 4, 4, 4, #damage multipliers
                1, 1, 1, 1, #stab flags
                1, 1, 1, 1, #is heal move
                1, 1, 1, 1, #is status move
                1, 1, 1, 1, #is boost move
                 1, 1,     #our and opponent remaining pokem
                 1, 1,      #our and opponent statused pokemon
                 1, 1,      #our and opponent fractional hp 
                 1,         #speed control
                 1, 1, 1, 1,  #priority values
                 1, 1, 1, 1,  #move category
                 1]          #Tera Type
        
        high = high + 14 * [1] #opponent and our boosts

        high = high + 6 * (2 * [1]) + 2 * [1] #typing

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

        # PokemonType -> embedding index
        self.type_to_idx = {ptype: i for i, ptype in enumerate(TYPE_ORDER)}

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Returns reward for current Battle State"""
        return self.reward_computing_helper(
            battle,
            fainted_value=1.0,
            hp_value= 0,
            status_value=0,
            victory_value=10.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Embeds our battle into an observation space"""
        moves_base_power, damage_multipliers, stab_flags, heal_flags, status_flags, boost_flags = self._Embed_moves(battle)

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

        #Embed our tera-type
        tera_type = self._embed_tera_type(battle)

        #Embed the stat boosts for each active pokemon
        our_boosts = self._embed_boosts(battle.active_pokemon)
        opp_boosts = self._embed_boosts(battle.opponent_active_pokemon)
    
        # Concatenate all features into a single observation vector
        observation = np.concatenate(
            [
                moves_base_power,
                damage_multipliers,
                stab_flags,
                heal_flags,
                status_flags,
                boost_flags,
                [our_remaining_pokemon],
                [opponent_remaining_pokemon],
                [our_statused_pokemon],
                [opponent_statused_pokemon],
                [our_active_fraction_hp],
                [opponent_active_fraction_hp],
                [speed_control],
                priority_values,
                move_categories,
                our_boosts,
                opp_boosts,
                [tera_type],
            ]
        )

        #Concatenate the typing observation for each of our pokemon
        for mon in battle.team.values():
            observation = np.concatenate([observation, self._embed_typing(mon)])

        #Concatenate the typing oservation for opponents active pokemon
        observation = np.concatenate([observation, self._embed_typing(battle.opponent_active_pokemon)])

        return observation.astype(np.float32)
    
    def _Embed_moves(self, battle: AbstractBattle) -> Tuple[np.ndarray, np.ndarray]:
        '''Helper function to embed move information'''
        moves_base_power = -np.ones(4)      # base power values (normalized)
        damage_multipliers = np.ones(4)     # type effectiveness multipliers
        stab_flags = np.zeros(4)            # same type attack bonus
        heal_flags = np.zeros(4)
        status_flags = np.zeros(4)
        boost_flags = np.zeros(4)

        all_moves = battle.active_pokemon.moves.values()

        # Enumerate through available moves and populate the vectors
        for i, move in enumerate(all_moves):
            if move not in battle.available_moves:
                continue

            # Normalize base power to be between 0 and 3 and update values. -1 if no move or move has no base power
            bp = move.base_power / 100 if move.base_power is not None else -1
            mult = move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
                type_chart=self.type_chart,
            )

            #update bp for each move
            moves_base_power[i] = bp
            # Update damage multiplier values for each move
            damage_multipliers[i] = mult
            #update stab flag for each move
            stab_flags[i] = 1 if move.type in battle.active_pokemon.types else 0
            #update heal flag for each move
            heal_flags[i] = 1 if ("drain" in move.entry) or ("heal" in move.entry) else 0
            #update status flag for each move
            status_flags[i] = 1 if "status" in move.entry else 0
            #update boost flag for each move
            boost_flags[i] = 1 if "selfBoost" in move.entry else 0

        
        return moves_base_power, damage_multipliers, stab_flags, heal_flags, status_flags, boost_flags
    
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
    
    def _embed_boosts(self, mon) -> np.ndarray:
        '''embeds boosts for pokemon'''
        boosts = np.zeros(7)

        #iterate through each stat
        for i, boost in enumerate(mon.boosts.values()):
            boosts[i] = boost / 7 #nomralize from -1 to 1
        
        return boosts
    
    def _embed_typing(self, mon) -> np.ndarray:
        '''embeds the active pokemon's typing for opponent'''
        typing_info = -np.ones(2)
        if mon.status == Status.FNT: 
            return typing_info
        types = mon.types

        for i, t in enumerate(types):
            idx = self.type_to_idx.get(t)
            if idx is not None:
                typing_info[i] = idx / len(TYPE_ORDER)

        return typing_info

    def _embed_tera_type(self, battle: AbstractBattle) -> int:
        '''Embeds the tera type of our active pokemon'''
        tera_type = battle.active_pokemon.tera_type
        if tera_type: 
            return self.type_to_idx.get(tera_type) / len(TYPE_ORDER)
        else:
            return -1


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

        force_switch = self.env.battle1.force_switch #Check force switch before we take action
        battle = self.env.battle1 #check for battle before we take action
        action = np.int64(action)
        #Proceed with the original step function
        actions = {
            self.env.agent1.username: action,
            self.env.agent2.username: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        #We will eventually remove this for true action masking.
        if is_invalid: #Give a negative penalty if move was selected outside of my action mask
            rewards[self.env.agent1.username] = rewards[self.env.agent1.username] - .5 
            #slight negative penalty for picking invalid option.

        '''if (action < 6 and not force_switch):
            rewards[self.env.agent1.username] = rewards[self.env.agent1.username] - .1'''
            #slight negative penalty for switching as agent has been switching too often.
        
        #Slight positive reward for using super effective moves
        '''if not force_switch and 5 < action:
                mvs = (
                    battle.available_moves
                    if len(battle.available_moves) == 1
                    and battle.available_moves[0].id in ["struggle", "recharge"]
                    else list(battle.active_pokemon.moves.values())
                )
                move = mvs[(action - 6) % len(mvs)]
                if move in (MoveCategory.PHYSICAL, MoveCategory.SPECIAL):
                    damage_multiplier = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=self.env.type_chart,
                )
                    if damage_multiplier >= 2:
                        rewards[self.env.agent1.username] = rewards[self.env.agent1.username] + .3
                    
                    elif damage_multiplier == 0:
                        rewards[self.env.agent1.username] = rewards[self.env.agent1.username] - .5

                    elif damage_multiplier < 1.0:
                        rewards[self.env.agent1.username] = rewards[self.env.agent1.username] - .25'''
        
        
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )