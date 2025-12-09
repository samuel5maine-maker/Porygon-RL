# Import necessary libraries
from poke_env.battle import AbstractBattle

from Gen9SinglesEnv.Gen9Env import Gen9SinglesEnv, MaskedSingleAgentWrapper

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from Agents.MaxDamage_Agent import MaxDamageAgent
from Agents.Random_Agent import RandomAgent
from Agents.Mixed_Agent import MixedAgent

import logging
logging.getLogger("poke-env").setLevel(logging.ERROR)

import os

def make_base_env():
    '''Builds our Generic Gen 9 Singles Environment'''
    base_env = Gen9SinglesEnv(
        battle_format="gen9randombattle",
        strict=False,
    )
    return base_env

def make_train_env(opponent):
    '''Builds our training environment'''
    # Instantiate a training environment
    base_env = make_base_env()

    opponent = opponent
    train_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

    #Wrap environment with a Monitor so we can log episode lengths and rewards
    train_env = Monitor(train_env)
    return train_env


def make_eval_env(opponent):
    '''Builds our evaluation environment'''
    #Instantiate evaluation environment
    base_env = make_base_env()

    opponent = opponent
    eval_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)
    
    #Wrap environment with a Monitor so we can log episode lengths and rewards
    eval_env = Monitor(eval_env)
    return eval_env
    

def test_environment(base_env):
    '''tests to make sure our environment is set up correctly'''
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

def build_model(train_env):
    '''Builds DQN model and returns the model'''
    policy_kwargs = dict(
        net_arch = [256, 256]
    )

    # Build the model: map observation space to action space
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=5e-5,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=.995,
        train_freq=4,
        exploration_fraction=0.20,
        exploration_final_eps=0.02,
        verbose=0,
        policy_kwargs=policy_kwargs
    )

    return model

def train_model(model, timesteps, reset_num_timesteps=True, log_interval=100):
    '''trains the model saves it then returns the model'''
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=reset_num_timesteps,
        log_interval=log_interval,
        progress_bar=True,
    )

    #Save the model
    model.save("dqn_pokemon_shodown_gen9_randombattles")

    return model

def train_in_chunks(
    model,
    total_timesteps: int,
    chunk_size: int,
    save_path: str,
    first_chunk_reset: bool = True,
    log_interval: int = 100,
):
    """
    Train `total_timesteps` in chunks of `chunk_size`, always overwriting `save_path`.
    Example: save_path="phase3_maxdamage_latest.zip"
    """
    timesteps_trained_before = model.num_timesteps
    remaining = total_timesteps

    first = True
    while remaining > 0:
        this_chunk = min(remaining, chunk_size)

        model.learn(
            total_timesteps=this_chunk,
            reset_num_timesteps=(first and first_chunk_reset),
            log_interval=log_interval,
            progress_bar=True,
        )

        remaining -= this_chunk
        first = False

        # Overwrite single checkpoint
        model.save(save_path)
        print(f"[CHECKPOINT] Saved model at {save_path} (global steps: {model.num_timesteps})")

    print(
        f"Finished chunked training: {model.num_timesteps - timesteps_trained_before} new steps "
        f"(total {model.num_timesteps})."
    )

    return model

def eval_model(eval_env, model):
    '''Evaluates the models performance'''
    #Unwrap to our base Gen9SinglesEnv
    base_env = eval_env.env.env

    #Record wins before evaluation
    wins_before = base_env.agent1.n_won_battles
    finished_before = base_env.agent1.n_finished_battles

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    #compute winrate for policy evaluation
    wins_after = base_env.agent1.n_won_battles
    finished_after = base_env.agent1.n_finished_battles

    wins = wins_after - wins_before
    finished= finished_after - finished_before

    winrate = wins/finished if finished > 0 else 0.0

    return mean_reward, std_reward, winrate
        
if __name__ == "__main__":
    # Create a base environment for Pokemon Single Battles
    base_env = make_base_env()

    test_environment(base_env) #Test our base env to ensure everything works as intended

    '''=========PHASE 1: TRAIN AGAINST RANDOM AGENT========='''
    '''train_env = make_train_env(opponent=RandomAgent())
    eval_env = make_eval_env(opponent=RandomAgent())

    model = build_model(train_env=train_env)

    train_in_chunks(
        model,
        total_timesteps= 100_000,
        chunk_size=50_000,
        save_path="phase_1_random_agent.zip",
        first_chunk_reset=True
    )

    mean_reward, std_reward, winrate = eval_model(eval_env=eval_env, model=model)

    print(f'Phase 1 Complete: opponent: RandomAgent')
    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}, Winrate: {winrate:.2f}')'''

    '''=========PHASE 2: TRAIN AGAINST MIXED AGENT========='''
        
    '''train_env = make_train_env(opponent=MixedAgent())
    eval_env = make_eval_env(opponent=MixedAgent())

    model.set_env(train_env)

    train_in_chunks(
            model,
            total_timesteps = 100_000,
            chunk_size = 50_000,
            save_path = "phase_2_mixed_agent.zip",
            first_chunk_reset=True
    )

    mean_reward, std_reward, winrate = eval_model(eval_env=eval_env, model=model)

    print(f'Phase 2 Complete: opponent: MixedAgent')
    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}, Winrate: {winrate:.2f}')'''

    '''=========PHASE 3: TRAIN AGAINST MAX DAMAGE AGENT========='''
    model = DQN.load('phase_3_max_damage_agent.zip')

    train_env = make_train_env(opponent=MaxDamageAgent())
    eval_env = make_eval_env(opponent=MaxDamageAgent())

    model.set_env(train_env)

    mean_reward, std_reward, winrate = eval_model(eval_env=eval_env, model=model)

    print(f'Phase 3 Started: opponent: MaxDamageAgent')
    print(f'Current Mean Reward: {mean_reward:.2f}, Current Std Reward: {std_reward:.2f}, Current Winrate: {winrate:.2f}')


    train_in_chunks(
        model,
        total_timesteps=1_000_000,
        chunk_size=50_000,
        save_path="phase_3_max_damage_agent.zip",
        first_chunk_reset=False
    )

    mean_reward, std_reward, winrate = eval_model(eval_env=eval_env, model=model)

    print(f'Phase 3 Complete: opponent: MaxDamageAgent')
    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}, Winrate: {winrate:.2f}')







    


