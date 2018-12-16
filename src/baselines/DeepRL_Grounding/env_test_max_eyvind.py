import vizdoom
import argparse
import env as grounding_env
import numpy as np
from utils.doom import *

parser = argparse.ArgumentParser(description='Grounding Environment Test')
parser.add_argument('-l', '--max-episode-length', type=int, default=300000,
                    help='maximum length of an episode (default: 30)')
parser.add_argument('-d', '--difficulty', type=str, default="hard",
                    help="""Difficulty of the environment,
                    "easy", "medium" or "hard" (default: hard)""")
parser.add_argument('--living-reward', type=float, default=0,
                    help="""Default reward at each time step (default: 0,
                    change to -0.005 to encourage shorter paths)""")
parser.add_argument('--frame-width', type=int, default=300,
                    help='Frame width (default: 300)')
parser.add_argument('--frame-height', type=int, default=168,
                    help='Frame height (default: 168)')
parser.add_argument('-v', '--visualize', type=int, default=1,
                    help="""Visualize the envrionment (default: 1,
                    change to 0 for faster training)""")
parser.add_argument('--sleep', type=float, default=0,
                    help="""Sleep between frames for better
                    visualization (default: 0)""")
parser.add_argument('-t', '--use_train_instructions', type=int, default=1,
                    help="""0: Use test instructions, 1: Use train instructions
                    (default: 1)""")
parser.add_argument('--scenario-path', type=str, default="maps/room.wad",
                    help="""Doom scenario file to load
                    (default: maps/room.wad)""")
parser.add_argument('--interactive', type=int, default=0,
                    help="""Interactive mode enables human to play
                    (default: 0)""")
parser.add_argument('--all-instr-file', type=str,
                    default="data/instructions_all.json",
                    help="""All instructions file
                    (default: data/instructions_all.json)""")
parser.add_argument('--train-instr-file', type=str,
                    default="data/instructions_train.json",
                    help="""Train instructions file
                    (default: data/instructions_train.json)""")
parser.add_argument('--test-instr-file', type=str,
                    default="data/instructions_test.json",
                    help="""Test instructions file
                    (default: data/instructions_test.json)""")
parser.add_argument('--object-size-file', type=str,
                    default="data/object_sizes.txt",
                    help='Object size file (default: data/object_sizes.txt)')

if __name__ == '__main__':
    args = parser.parse_args()
    env = grounding_env.GroundingEnv(args)
    env.game_init()

    num_episodes = 0
    rewards_per_episode = []
    reward_sum = 0
    is_final = 1
    while num_episodes < 100:
        if is_final:
            (image, instruction), _, _, _ = env.reset()
            print("Instruction: {}".format(instruction))

        print("player angle", get_agent_orientation(env.game))

        print("x,y", get_agent_location(env.game))

        pos_goal = []
        pos_goal.append(env.object_coordinates[env.correct_location].x)
        pos_goal.append(env.object_coordinates[env.correct_location].y)
        pos = get_agent_location(env.game)
        angle = get_agent_orientation(env.game)


        # compute the angle between the players orientation and the goal in [0,180]

        # get unit vector for player orientation

        angle = angle/(180/np.pi)
        p_unit = (np.cos(angle), np.sin(angle))
        # print("p_unit", p_unit)
        g_unit = (np.array(pos_goal) - np.array(pos))
        g_unit = g_unit/np.linalg.norm(g_unit)
        # print("g_unit", g_unit)

        theta=np.arccos(np.dot(p_unit, g_unit))
        print("angle between player view and goal should be ", theta)
        rdist = grounding_env.get_l2_distance(pos[0], pos[1], pos_goal[0], pos_goal[1])
        print("distance between player view and goal should be ", rdist)

        # Take a random action
        (image, instruction), reward, is_final, _ = \
            env.step(np.random.randint(3))
        reward_sum += reward

        if is_final:
            print("Total Reward: {}".format(reward_sum))
            rewards_per_episode.append(reward_sum)
            num_episodes += 1
            reward_sum = 0
            if num_episodes % 10 == 0:
                print("Avg Reward per Episode: {}".format(
                    np.mean(rewards_per_episode)))
