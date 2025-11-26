#Max Damage Agent For Pokemon Showdown
from poke_env import Player
import sys
import asyncio

class MaxDamageAgent(Player):
    def choose_move(self, battle):
        # Select the move that deals the maximum damage

        if battle.available_moves:
            #Get the move with the highest base power
            best_move = max(battle.available_moves, key=lambda move: move.base_power if move.base_power is not None else 0)
            #Create and return the order for that move
            return self.create_order(best_move)
        else:
            #Otherwise perform a random move (should rarely happen)
            return self.choose_random_move(battle)
        


# Example usage:
async def main():
    #Initialize two Max Damage Agents
    max_damage_agent1 = MaxDamageAgent()
    max_damage_agent2 = MaxDamageAgent()

    #Start a battle between the two Max Damage Agents
    await max_damage_agent1.battle_against(max_damage_agent2, n_battles=1)

    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())

