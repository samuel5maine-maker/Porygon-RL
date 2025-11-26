"""
Reinforcement Learning Agent Module

Tested with:
- Python 3.10.11
- poke-env 0.5.10
- stable-baselines3 1.8.0
- gymnasium 0.28.1
"""

# Import necessary libraries
from typing import Any, Awaitable, Dict, Optional, Tuple
import sys
import asyncio

import numpy as np

from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.battle import AbstractBattle
from poke_env.environment.env import ActionType, ObsType, PokeEnv
from poke_env.battle.status import Status
from poke_env.battle.move_category import MoveCategory

from gymnasium.utils.env_checker import check_env
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from Random_Agent import RandomAgent
from MaxDamage_Agent import MaxDamageAgent


from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


# Define Type Chart for Damage Multipliers
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
        #20-45 entries are all for action mask
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, -1, -1, -1, -1, -1,
                -1, -1, -1] + [0] * 26
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1] + [1] * 26

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
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Embeds our battle into an observation space"""
        moves_base_power = -np.zeros(4)      # base power values (normalized)
        damage_multipliers = np.zeros(4)     # type effectiveness multipliers

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

        # Count how many Pokemon are fainted on each side (normalized 0–1)
        our_remaining_pokemon = len(
            [mon for mon in battle.team.values() if not mon.fainted]
        ) / 6
        opponent_remaining_pokemon = len(
            [mon for mon in battle.opponent_team.values() if not mon.fainted]
        ) / 6

        #Count how many Pokemon are statused on each side (normalized 0–1)
        our_statused_pokemon = len(
            [mon for mon in battle.team.values() if mon.status is not None and not mon.fainted]
        ) / 6

        #Count how many Pokemon are statused on opponent's side (normalized 0–1)
        opponent_statused_pokemon = len(
            [mon for mon in battle.opponent_team.values() if mon.status is not None and not mon.fainted]
        ) / 6

        #Get fraction hp of our active pokemon (normalized 0–1)
        our_active_fraction_hp = battle.active_pokemon.current_hp_fraction

        #Get fraction hp of opponent's active pokemon (normalized 0–1)
        opponent_active_fraction_hp = battle.opponent_active_pokemon.current_hp_fraction

        #Encoding speed control (1 if we are faster, 0 if slower)
        our_speed = battle.active_pokemon.base_stats["spe"] * (1 + battle.active_pokemon.boosts["spe"] * .5)
        opponent_speed = battle.opponent_active_pokemon.base_stats["spe"] * (1 + battle.opponent_active_pokemon.boosts["spe"] * .5)

        if battle.active_pokemon.status == Status.PAR:
            our_speed = our_speed * .5
        if battle.opponent_active_pokemon.status == Status.PAR:
            opponent_speed = opponent_speed * .5
        speed_control = 1 if our_speed > opponent_speed else 0

        #Encode priority of each move
        priority_values = np.zeros(4)
        for i, move in enumerate(battle.available_moves):
            if move._id in self.special_moves:
                priority_values[i] = 0  # Assign neutral priority for special moves
            else:
                priority_values[i] = move.priority / 7  # Normalize priority to be between -1 and 1

        #Encode type (physical/special/status) of each move
        move_categories = np.zeros(4)
        for i, move in enumerate(battle.available_moves):
            if move.category == MoveCategory.PHYSICAL:
                move_categories[i] = 1
            elif move.category == MoveCategory.SPECIAL:
                move_categories[i] = 0
            else:
                move_categories[i] = -1

        #Embed the current action mask into the observation
        action_mask = self.get_action_mask(battle)
    
        # Concatenate all features into a single observation vector
        observation = np.concatenate(
            [
                moves_base_power,
                damage_multipliers,
                [our_remaining_pokemon],
                [opponent_remaining_pokemon],
                [our_statused_pokemon],
                [opponent_statused_pokemon],
                [our_active_fraction_hp],
                [opponent_active_fraction_hp],
                [speed_control],
                priority_values,
                move_categories,
                action_mask,
            ]
        ).astype(np.float32)

        return observation
    
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
        if action_mask[action] == 0:
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
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )
        

# Create a base environment for Pokemon Single Battles
base_env = Gen9SinglesEnv(
    battle_format="gen9randombattle",
    strict=False,
)
opponent = MaxDamageAgent()
# Right now opponent is a Random Agent, will train against another RL Agent later.

# Wrap it as a Gymnasium Environment
test_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

# Check our environment is valid
obs, info = test_env.reset()

done = False
truncated = False
while not done and not truncated:
    action = test_env.action_space.sample()
    obs, reward, done, truncated, info = test_env.step(action)
    test_env.env.get_action_mask(test_env.env.battle1)

test_env.close()

# Instantiate a training and evaluation environment
opponent = MaxDamageAgent()
train_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

opponent = MaxDamageAgent()
eval_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

#Wrap environments with a Monitor so we can log episode lengths and rewards
train_env = Monitor(train_env)
eval_env = Monitor(eval_env)

# Build the model: map observation space to action space
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=20_000,
    batch_size=32,
    gamma=.99,
    train_freq=4,
    exploration_fraction=0.25,
    exploration_final_eps=0.05,
    verbose=1,
)

# Train the model
model.learn(
    total_timesteps=1_200_000,
    log_interval=100,
    progress_bar=True,
)

#Save the model
model.save("dqn_pokemon_shodown_gen9_randombattles")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=20,
    deterministic=True,
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

#Watch a full battle
obs, info = eval_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

eval_env.close()



