import asyncio
import sys
import random
from poke_env.player.env_player import EnvPlayer, AccountConfiguration
from poke_env.data import GenData
from gymnasium.spaces import Box
import numpy as np
from poke_env.player.random_player import RandomPlayer
from poke_env.player.env_player import Gen9EnvSinglePlayer
from poke_env.environment.double_battle import DoubleBattle
from gym.spaces import Discrete
from poke_env.player.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, DefaultBattleOrder
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate

sys.path.append(".")  # will make "utils" callable from root
sys.path.append("..")  # will make "utils" callable from simulators

from helpers.doubles_utils import *

GEN_9_DATA = GenData.from_gen(9)

class DoubleBattleRLPlayer(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def describe_embedding(self):
        """
        Define the action space for the agent in double battles.
        """
        max_moves = 4  # Maximum available moves per Pokémon
        max_switches = 6  # Maximum possible switches per Pokémon
        # Total possible actions: moves and switches for both Pokémon
        return Discrete((max_moves + max_switches) * 2)

    def action_to_move(self, action: int, battle: DoubleBattle):
        """
        Translate an action into a set of orders for the active Pokémon.
        """
        # Available moves and switches for each active Pokémon
        available_moves_p1 = battle.available_moves[0] if battle.active_pokemon[0] else []
        available_moves_p2 = battle.available_moves[1] if battle.active_pokemon[1] else []
        available_switches_p1 = battle.available_switches[0] if battle.active_pokemon[0] else []
        available_switches_p2 = battle.available_switches[1] if battle.active_pokemon[1] else []

        # Number of available moves and switches for each Pokémon
        num_moves_p1 = len(available_moves_p1)
        num_moves_p2 = len(available_moves_p2)
        num_switches_p1 = len(available_switches_p1)
        num_switches_p2 = len(available_switches_p2)

        # Determine action for the first Pokémon
        if action < num_moves_p1:
            # Use move
            move = available_moves_p1[action]
            target = battle.get_possible_showdown_targets(move, battle.active_pokemon[0])[0]
            move_order_p1 = self.create_order(move, target=target)
        elif action < num_moves_p1 + num_switches_p1:
            # Switch Pokémon
            switch_index = action - num_moves_p1
            move_order_p1 = self.create_order(available_switches_p1[switch_index])
        else:
            # Invalid action: choose random move
            move_order_p1 = self.choose_random_move(battle)

        # Determine action for the second Pokémon
        action_p2 = action - (num_moves_p1 + num_switches_p1)
        if action_p2 < num_moves_p2:
            # Use move
            move = available_moves_p2[action_p2]
            target = battle.get_possible_showdown_targets(move, battle.active_pokemon[1])[0]
            move_order_p2 = self.create_order(move, target=target)
        elif action_p2 < num_moves_p2 + num_switches_p2:
            # Switch Pokémon
            switch_index = action_p2 - num_moves_p2
            move_order_p2 = self.create_order(available_switches_p2[switch_index])
        else:
            # Invalid action: choose random move
            move_order_p2 = self.choose_random_move(battle)

        # Return orders for both Pokémon
        return move_order_p1, move_order_p2

    def embed_battle(self, battle: DoubleBattle):
        """
        Custom embedding of the battle state.
        """
        # Here you can add a numerical representation of the battle state
        # For now, returning a placeholder
        return np.zeros(100)  # Placeholder for a valid representation

    def calc_reward(self, battle: DoubleBattle) -> float:
        """
        Calculate the reward based on the current battle state.
        """
        reward = 0

        # Reward for defeating opponent's Pokémon
        reward += (6 - len(battle.opponent_team)) * 10

        # Penalty for losing own Pokémon
        reward -= (6 - len(battle.team)) * 10

        # Reward for inflicting damage
        if battle.opponent_active_pokemon:
            reward += sum(
                [1 - mon.current_hp_fraction for mon in battle.opponent_active_pokemon if mon]
            )

        # Penalty for receiving damage
        if battle.active_pokemon:
            reward -= sum(
                [1 - mon.current_hp_fraction for mon in battle.active_pokemon if mon]
            )

        # Minimal penalty to encourage continuous action
        reward -= 0.1

        return reward

    def handle_popup_message(self, message, battle):
        if "There's already a challenge" in message:
            self.cancel_challenge(battle)

    def cancel_challenge(self, battle):
        self.send_message("/forfeit", battle.battle_tag)

    async def _handle_message(self, message_tree):
        if message_tree.startswith("|popup|"):
            self.handle_popup_message(message_tree, self.battle)
        await super()._handle_message(message_tree)

class RandomDoublesPlayer(Player):
    def print_message(self, msg, battle):
        asyncio.ensure_future(self._send_message(msg, battle.battle_tag))

    def choose_move(self, battle):
        orders = self.get_all_doubles_moves(battle)
        filtered_orders = list(filter(lambda x: DoubleBattleOrder.is_valid(battle, x), orders))
        if filtered_orders:
            order = random.choice(filtered_orders)
        else:
            order = DefaultBattleOrder()
        return order

    def teampreview(self, battle):
        # We use 1-6 because showdown's indexes start from 1
        return "/team " + "".join(random.sample(list(map(lambda x: str(x + 1), range(0, len(battle.team)))), k=4))

# Define a valid team for the battle format
team = """
Garchomp @ Choice Scarf
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Scale Shot
- Earthquake
- Stealth Rock

Dragonite @ Leftovers
Ability: Multiscale
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Dragon Dance
- Outrage
- Fire Punch
- Roost

Tyranitar @ Weakness Policy
Ability: Sand Stream
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Stealth Rock
- Crunch
- Earthquake
- Ice Punch

Talonflame @ Life Orb
Ability: Gale Wings
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Brave Bird
- Flare Blitz
- Swords Dance
- Roost

Rotom-Wash @ Sitrus Berry
Ability: Levitate
EVs: 252 HP / 4 SpA / 252 SpD
Sassy Nature
- Hydro Pump
- Volt Switch
- Will-O-Wisp
- Pain Split

Gyarados @ Aspear Berry
Ability: Intimidate
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Dance
- Waterfall
- Earthquake
- Ice Fang
"""

# Initialize the opponent and the environment player
opponent = RandomDoublesPlayer(
    account_configuration=AccountConfiguration(username="randRoberto2", password="pass")
)

env_player = DoubleBattleRLPlayer(
    opponent=opponent,
    account_configuration=AccountConfiguration(username="DoublesAgent2", password="pass"),
    battle_format="gen9vgc2024regh",  # Battle format for VGC
    team=team  # Provide the valid team
)

# Function to run the environment and let the agent interact with it
async def main():
    print("\033[92m Starting script... \033[0m")

    # We create players:
    players = [
        env_player,
        opponent,
    ]

    # Each player plays n times against each other
    n = 5

    # Pit players against each other
    print("About to start " + str(n * sum(i for i in range(0, len(players)))) + " battles...")
    cross_evaluation = await cross_evaluate(players, n_challenges=n)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))

if __name__ == "__main__":
    asyncio.run(main())