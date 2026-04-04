from customer_support_env.server.customer_support_env_environment import CustomerSupportEnvironment

env = CustomerSupportEnvironment()

obs = env.reset()

print("User Query: ", obs.user_query)
print("Sentiment: ", obs.sentiment)
print("Issue Type: ", obs.issue_type)
print("Order Status: ", obs.order_status)
print("Attempts: ", obs.attempts)

current_state = env.state

print("Episode ID: ", current_state.episode_id)
print("Step Count: ", current_state.step_count)