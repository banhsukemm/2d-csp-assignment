import time
import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy2, RandomPolicy2
from student_submissions.s2210xxx.policy2210xxx import Policy2314047
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
if __name__ == "__main__":
    observation, info = env.reset(seed=2)

    policy2210xxx = Policy2314047()
    for __ in range(1): # run only 1 env for comparing 2 Policy
        start_time = time.time()
        curren_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        current_info = info
        stock_idx = -1
        action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        while True:
            last_action = action
            action = policy2210xxx.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            if (info["filled_ratio"] - current_info["filled_ratio"] > 0):
                if stock_idx > -1:
                    print("Used stock index:",last_action["stock_idx"],f",trim loss: {current_info["trim_loss"]:.4f}")
                stock_idx += 1
            current_info = info
            if terminated or truncated:
                print("Used stock index:",action["stock_idx"],f",trim loss: {current_info["trim_loss"]:.4f}")
                elapsed_time = time.time() - start_time
                print(f"Elapsed time: {elapsed_time:.4f} seconds")
                print("Total stock used:", stock_idx + 1)
                observation, info = env.reset()
                break


env.close()
