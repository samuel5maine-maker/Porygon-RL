Pokémon battles are a hard RL problem: the state space is enormous, information is imperfect (the opponent's team is partially hidden), and only a subset of actions is legal on any given turn. This project frames a Pokémon battle as a [Gymnasium](https://gymnasium.farama.org/) environment and trains a **Deep Q-Network (DQN)** to win battles through a **curriculum of progressively stronger scripted opponents**:

1. **Phase 1 — RandomAgent:** an opponent that picks uniformly random legal moves.
2. **Phase 2 — MixedAgent:** randomly plays as either the Random or Max-Damage agent each battle.
3. **Phase 3 — MaxDamageAgent:** a greedy opponent that always selects its highest base-power move.

After Phase 1 (100k timesteps), the agent reached an **85% win rate against the random opponent** (see `train.log`); later phases continue training the same network against harder opponents.

## Approach

- **Environment** (`Gen9SinglesEnv/Gen9Env.py`): subclasses poke-env's `SinglesEnv`. The battle state is embedded as a ~66-dimensional feature vector: for each of the active Pokémon's four moves, its normalized base power, type-effectiveness multiplier against the opponent, STAB/heal/status/boost flags, priority, and category; plus team-level features (remaining and statused Pokémon on both sides, active Pokémon HP fractions, speed comparison, stat boosts, Tera type, and team typing).
- **Action space:** 26 discrete actions (6 switches, 4 moves, and move variants for mega/Z/Dynamax/Terastallize; only switches, moves, and Tera are enabled in this format).
- **Action masking** (`MaskedSingleAgentWrapper`): the environment computes a legality mask each turn. If the agent picks an illegal action, a random legal action is substituted and a −0.5 reward penalty is applied, teaching the network to avoid illegal choices.
- **Reward:** +1 per opposing Pokémon fainted (−1 for own), +10 for winning the battle.
- **Model:** DQN with a 256×256 MLP policy (γ = 0.995, replay buffer of 300k, ε-greedy exploration annealed to 0.02), trained in checkpointed chunks so long runs can be resumed.

## Repository layout

| Path | Description |
|---|---|
| `PorygonRL.py` | Main training script: builds the environment, defines/trains the DQN, runs the three curriculum phases, and evaluates win rate. |
| `Gen9SinglesEnv/Gen9Env.py` | Custom Gymnasium environment: state embedding, reward function, action masking. |
| `Agents/` | Scripted baseline opponents: `Random_Agent.py`, `MaxDamage_Agent.py`, `Mixed_Agent.py`. |
| `Agent_PlayGround.py` | Debugging harness that verifies the action mask agrees with the engine's legality checks. |
| `phase_*_agent.zip`, `dqn_pokemon_shodown_gen9_randombattles.zip` | Saved model checkpoints from each training phase. |
| `Porygon-RL_*.docx` / `.pptx` | Final project write-up and presentation slides. |

## Running it

Requires Python 3.10+, a local [Pokémon Showdown server](https://github.com/smogon/pokemon-showdown) (poke-env battles against `localhost`), and the packages in `requirements.txt` (chiefly `poke-env`, `stable-baselines3`, `gymnasium`, `torch`).

```bash
pip install -r requirements.txt
# In a separate terminal, start a local Showdown server:
#   git clone https://github.com/smogon/pokemon-showdown && cd pokemon-showdown
#   node pokemon-showdown start --no-security
python PorygonRL.py
```

Training progress, evaluation rewards, and win rates are printed to the console; checkpoints are saved after every 50k-step chunk.
