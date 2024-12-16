from poke_env.player.env_player import EnvPlayer, AccountConfiguration, Gen9EnvSinglePlayer
from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder, DoubleBattleOrder
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.data import GenData
from gymnasium.spaces import Box
import numpy as np
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.double_battle import DoubleBattle
from gym.spaces import Discrete
from stable_baselines3 import A2C
import asyncio
import sys
import random
import time


GEN_9_DATA = GenData.from_gen(9)

class DoubleBattleRLPlayer(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def action_to_move(self, action: int, battle: DoubleBattle) -> BattleOrder:
        print(battle.active_pokemon)
        """
        Convierte las acciones a órdenes de movimiento específicas para batallas dobles, considerando la selección de objetivos.

        :param action: El índice de la acción a convertir.
        :param battle: La batalla en la que actuar.
        :return: La orden correspondiente para enviar al servidor.
        """
        
        targets = battle.opponent_active_pokemon
        #print(f"Targets: {targets}")
        switches = battle.available_switches[0] if len(battle.available_switches) > 0 else battle.available_switches[1]
        #print(f"Switches: {switches}")
        #print(f"Action: {action}, Available moves: {battle.available_moves}, Can Tera: {battle.can_tera}")
        if action == -1:
            # Si la acción es -1, se forfeit la batalla.
            return ForfeitBattleOrder()

        # Determina si el agente debe elegir el movimiento y el objetivo
        print(f"action: {action}, action < 4: {action < 4}, len(battle.available_moves): {len(battle.available_moves[0])}, action < len(battle.available_moves) : {action < len(battle.available_moves[0])}, battle.force_switch: {battle.force_switch}")
        if action < 4 and action < len(battle.available_moves[0]) and not any(battle.force_switch):
            print("Action is a move")
            # Selección del objetivo para el movimiento
            if battle.available_moves[0] != []:
                print(f"Moves : {battle.available_moves[0]}")
                move = battle.available_moves[0][action % len(battle.available_moves[0])]
                target = np.random.choice([1,2])
                first_order = self.agent.create_order(order = move, move_target=target)
            else:
                first_order = self.agent.choose_random_move(battle)

            if battle.available_moves[1] != []:
                print(f"Moves: {battle.available_moves[1]}")
                move = battle.available_moves[1][action % len(battle.available_moves[1])]
                target = np.random.choice([1,2])
                second_order = self.agent.create_order(order = move, move_target=target)
            else:
                return first_order
            print(f"First order: {first_order}, Second order: {second_order}")
            order = DoubleBattleOrder(first_order, second_order)
            print(order)
            return order
        if 0 <= action - 16 < len(battle.available_moves[0]) and battle.can_tera and not battle.force_switch:
            print(f"Action is a tera: {action-16}")
            # Selección del objetivo para el movimiento
            move = battle.available_moves[0][action - 16]
            # En batallas dobles, el movimiento debe dirigirse a uno de los Pokémon enemigos
            target = np.random.choice([1,2])  # Aquí elegimos el Pokémon enemigo activo 
            first_order = self.agent.create_order(order = move, move_target=target, terastallize=True)
            move = battle.available_moves[1][action - 16]
            target = np.random.choice([1,2])
            second_order = self.agent.create_order(order = move, move_target=target)
            print(f"First order: {first_order}, Second order: {second_order}")
            return DoubleBattleOrder(first_order, second_order)

        # Si la acción está dentro del rango de cambios de Pokémon
        elif 0 <= action - 20 < len(switches):
            print(f"Action is a switch: {action-20}")
            # Cambiar un Pokémon disponible, en batallas dobles solo se puede cambiar un Pokémon
            target = switches[action - 20]
            print(target)
            return self.agent.create_order(target)

        # Si la acción no es válida, elegir un movimiento aleatorio
        else:
            print("Action is invalid")
            return self.agent.choose_random_doubles_move(battle)
            '''
            # Aquí se hace una selección aleatoria de un movimiento
            move = battle.available_moves[0][random.randint(0, len(battle.available_moves[0]) - 1)]
            # En batallas dobles, se selecciona el objetivo aleatorio también
            target = np.random.choice([1,2])
            first_order = self.agent.create_order(order= move, move_target=target)
            move = battle.available_moves[1][random.randint(0, len(battle.available_moves[1]) - 1)]
            target = np.random.choice([1,2])
            second_order = self.agent.create_order(order= move, move_target=target)
            print(f"First order: {first_order}, Second order: {second_order}")
            order = DoubleBattleOrder(first_order, second_order)
            print(order)
            
            
            return order'''

    def embed_battle(self, battle: DoubleBattle):
        """
        Custom embedding of the battle state for Double Battles.
        """
        # Get HP fraction of active Pokémon, safely handling None
        hp_p1a = battle.active_pokemon[0].current_hp_fraction if battle.active_pokemon[0] else 0.0
        hp_p1b = battle.active_pokemon[1].current_hp_fraction if len(battle.active_pokemon) > 1 and battle.active_pokemon[1] else 0.0

        # Get HP fraction of opponent Pokémon, safely handling None
        hp_p2a = battle.opponent_active_pokemon[0].current_hp_fraction if battle.opponent_active_pokemon[0] else 0.0
        hp_p2b = battle.opponent_active_pokemon[1].current_hp_fraction if len(battle.opponent_active_pokemon) > 1 and battle.opponent_active_pokemon[1] else 0.0

        # Moves available for active Pokémon, handling potential None
        moves_p1a = len(battle.available_moves[0]) if battle.available_moves and battle.available_moves[0] else 0
        moves_p1b = len(battle.available_moves[1]) if len(battle.available_moves) > 1 and battle.available_moves[1] else 0

        return np.array([
            hp_p1a,  # HP of first active Pokémon
            hp_p1b,  # HP of second active Pokémon
            hp_p2a,  # HP of opponent's first active Pokémon
            hp_p2b,  # HP of opponent's second active Pokémon
            moves_p1a,  # Moves available for Pokémon 1
            moves_p1b,  # Moves available for Pokémon 2
            battle.turn,  # Turn number
            0,  # Placeholder for additional features
            0,
            0
        ])


    
    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )



class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.agent.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

opponent = RandomPlayer(account_configuration=AccountConfiguration(username="randRoberto", password="pass"), battle_format="gen9randomdoublesbattle")
env_player = DoubleBattleRLPlayer(
    opponent=opponent,
    account_configuration=AccountConfiguration(username="DAgentIAJuegos", password="pass"),
    battle_format="gen9randomdoublesbattle"  # Formato de batalla para VGC
)
second_opponent = MaxDamagePlayer(account_configuration=AccountConfiguration(username="MaxDamAAA", password="a"), battle_format="gen9randomdoublesbattle")


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

np.random.seed(0)


model_store = {}

# This is the function that will be used to train the a2c
def a2c_training(player, nb_steps):
    model = A2C("MlpPolicy", player, verbose=1)
    model.learn(total_timesteps=10_000)
    model_store[player] = model
    


def a2c_evaluation(player, nb_episodes):
    # Reset battle statistics
    model = model_store[player]
    player.reset_battles()
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )



NB_TRAINING_STEPS = 100
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)

if __name__ == "__main__":

    model = A2C("MlpPolicy", env_player, verbose=1)
    model.learn(total_timesteps=NB_TRAINING_STEPS)
    print("Training finished")

    obs, reward, done, _, info = env_player.step(0)
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)
    print("First opponent won", env_player.n_won_battles, "battles")


    finished_episodes = 0
    
    env_player.reset_env()
    obs, _ = env_player.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)

    finished_episodes = 0
    env_player._opponent = second_opponent

    # Ensure all battles are finished before resetting
    while env_player.battles:
        env_player.complete_current_battle()

    env_player.reset_battles()
    obs, _ = env_player.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)