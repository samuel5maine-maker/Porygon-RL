'''Playground for testing and debugging agents'''
import numpy as np

from Gen9SinglesEnv.Gen9Env import Gen9SinglesEnv, MaskedSingleAgentWrapper
from Agents.Random_Agent import RandomAgent

if __name__ == "__main__":
    base_env = Gen9SinglesEnv(
        battle_format="gen9randombattle",
        strict=False,
    )

    opponent = RandomAgent()
    playground_env = MaskedSingleAgentWrapper(base_env, opponent=opponent)

    obs, info = playground_env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        battle = playground_env.env.battle1
        mask = playground_env.env.get_action_mask(battle)

        # Test every action 0..25
        for action in range(26):
            try:
                playground_env.env.action_to_order(
                    np.int64(action),
                    battle,
                    fake=False,    # let legality checks run
                    strict=True,
                )
                legal = True
            except AssertionError as e:
                if str(e) == "invalid action":
                    legal = False
                else:
                    raise    # any other assertion is unexpected

            if legal and mask[action] == 0:
                print(f"BUG: Action {action} is legal but mask[action]==0 on turn {battle.turn}")
            if (not legal) and mask[action] == 1:
                print(f"BUG: Action {action} is illegal but mask[action]==1 on turn {battle.turn}")

        # now actually play one valid action to advance the battle
        valid_actions = [i for i, v in enumerate(mask) if v == 1]
        if valid_actions:
            chosen = int(np.random.choice(valid_actions))
        else:
            chosen = int(playground_env.action_space.sample())

        obs, reward, done, truncated, info = playground_env.step(chosen)

    print("Battle finished")
    playground_env.close()
