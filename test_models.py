from customer_support_env.models import Observation, Action

# Create observation
obs = Observation(
    user_query="Where is my order?",
    sentiment="angry",
    issue_type="delivery",
    order_status="delayed",
    attempts=1
)

# Create action
act = Action(action="apologize")

print("Observation:", obs)
print("Action:", act)