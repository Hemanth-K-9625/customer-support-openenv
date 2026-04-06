from customer_support_env.server.customer_support_env_environment import CustomerSupportEnvironment

env = CustomerSupportEnvironment()


# ----------------------------
# 🔹 EASY TASK TEST
# ----------------------------
print("---- EASY TASK ----")
obs = env.reset(task="easy")
print("Initial:", obs)

obs, reward, done, info = env.step("refund")
print("Action: refund | Reward:", reward, "| Done:", done)
print("Info:", info)


# ----------------------------
# 🔹 MEDIUM TASK TEST
# ----------------------------
print("\n---- MEDIUM TASK ----")
obs = env.reset(task="medium")
print("Initial:", obs)

actions = ["apologize", "apologize", "track_order", "refund"]

for action in actions:
    obs, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward} | Done: {done}")
    print("State:", obs)

    if done:
        print("Episode ended:", info.get("termination_reason"))
        break


# ----------------------------
# ❌ INVALID ACTION TEST
# ----------------------------
print("\n---- INVALID ACTION TEST ----")
env.reset(task="medium")

obs, reward, done, info = env.step("random_action")

print("Action: random_action")
print("Reward:", reward)   # should be -0.2
print("Done:", done)
print("Info:", info)


# ----------------------------
# 🔁 REPEATED ACTION TEST
# ----------------------------
print("\n---- REPEATED ACTION TEST ----")
env.reset(task="medium")

actions = ["apologize", "apologize", "apologize"]

for action in actions:
    obs, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward}")


# ----------------------------
# 🌀 NO PROGRESS TEST
# ----------------------------
print("\n---- NO PROGRESS TEST ----")
env.reset(task="medium")

actions = ["ask_info", "ask_info", "ask_info"]

for action in actions:
    obs, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward}")


# ----------------------------
# 🔁 LOOP TEST (A → B → A → B)
# ----------------------------
print("\n---- LOOP TEST ----")
env.reset(task="medium")

actions = ["apologize", "provide_status_update"] * 3

for action in actions:
    obs, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward}")


# ----------------------------
# 🏁 COMPLETION TEST
# ----------------------------
print("\n---- COMPLETION TEST ----")
env.reset(task="easy")

actions = ["refund"]

for action in actions:
    obs, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward} | Done: {done}")
    print("Termination:", info.get("termination_reason"))


# ----------------------------
# ⛔ STRESS TEST (MAX STEPS)
# ----------------------------
print("\n---- STRESS TEST ----")
env.reset(task="hard")

for i in range(15):  # more than MAX_STEPS
    obs, reward, done, info = env.step("ask_info")
    print(f"Step {i} | Reward: {reward} | Done: {done}")

    if done:
        print("Stopped due to:", info.get("termination_reason"))
        break