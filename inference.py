from customer_support_env.server.customer_support_env_environment import (
    CustomerSupportEnvironment,
)


env = CustomerSupportEnvironment()
obs = env.reset(task="medium")
total_reward = 0.0
reward_history = []

for step in range(10):
    if step == 0:
        action = "apologize"
    elif step == 1:
        action = "track_order"
    else:
        action = "refund"

    obs, reward, done, info = env.step(action)
    total_reward += reward
    reward_history.append(reward)
    print(f"Step {step + 1} | action={action} | reward={reward} | done={done}")

    if done:
        break

total_reward = sum(reward_history)
normalized_score = max(0.0, min(1.0, total_reward / 10))

print(f"Total reward: {total_reward}")
print(f"Reward history: {reward_history}")
print(f"Normalized Score: {normalized_score}")
print("Episode finished")
