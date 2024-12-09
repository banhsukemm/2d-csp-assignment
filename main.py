import time
import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy2, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2314047, Policy2210xxx
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    observation, info = env.reset(seed=40)
    print(info)

    policy2210xxx = Policy2314047()
    # for _ in range(100):
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()
    for __ in range(3):
        demands = 0
        for prod in observation["products"]:
            demands += prod["quantity"]
        print("Demands: ", demands)
        print(observation)
        for _ in range(demands):
            action = policy2210xxx.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            print(info)

            if terminated or truncated:
                observation, info = env.reset()
                break


env.close()
