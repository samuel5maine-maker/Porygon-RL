# Import necessary libraries
from Gen9SinglesEnv.Gen9Env import Gen9SinglesEnv, MaskedSingleAgentWrapper

from Random_Agent import RandomAgent
from MaxDamage_Agent import MaxDamageAgent

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

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

def train_model(base_env):
    '''Builds and trains DQN model and returns the trained model'''
    # Instantiate a training environment
    opponent = MaxDamageAgent()
    train_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

    #Wrap environment with a Monitor so we can log episode lengths and rewards
    train_env = Monitor(train_env)

    # Build the model: map observation space to action space
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=.99,
        train_freq=4,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        verbose=1,
    )

    # Train the model
    model.learn(
        total_timesteps=300_000,
        log_interval=100,
        progress_bar=True,
    )

    #Save the model
    model.save("dqn_pokemon_shodown_gen9_randombattles")

    return model

def eval_model(base_env, model):
    '''Evaluates the models performance'''
    #Instantiate evaluation environment
    opponent = MaxDamageAgent()
    eval_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)
    
    #Wrap environment with a Monitor so we can log episode lengths and rewards
    eval_env = Monitor(eval_env)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    return mean_reward, std_reward
        
if __name__ == "__main__":
    # Create a base environment for Pokemon Single Battles
    base_env = Gen9SinglesEnv(
        battle_format="gen9randombattle",
        strict=False,
    )

    test_environment(base_env)
    model = train_model(base_env)
    mean_reward, std_reward = eval_model(base_env, model)

    print(f'Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}')



