#Import necessary modules
import sys
import asyncio
from poke_env import Player



#Our Random Agent
class RandomAgent(Player):
    def choose_move(self, battle) -> int:
        if battle.won: print(battle.won)

        return self.choose_random_move(battle)
    
    
    

async def main():
    #Start a battle between the two random agents
    #Initialize two random agents with random teams
    random_player1 = RandomAgent()
    random_player2 = RandomAgent()

    await random_player1.battle_against(random_player2, n_battles=10)
    

    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())