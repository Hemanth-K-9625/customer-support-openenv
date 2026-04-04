from pydantic import BaseModel

# Observation Model (required for your task)
class Observation(BaseModel):
    user_query: str
    sentiment: str
    issue_type: str
    order_status: str
    attempts: int


# Action Model (required for your task)
class Action(BaseModel):
    action: str


# 🔥 Aliases for OpenEnv compatibility
class CustomerSupportObservation(Observation):
    pass


class CustomerSupportAction(Action):
    pass