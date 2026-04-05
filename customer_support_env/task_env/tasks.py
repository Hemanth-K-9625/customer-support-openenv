"""Task definitions for the customer support environment."""


class CustomerSupportTask:
    def __init__(self, initial_state, goal: str):
        self.initial_state = initial_state
        self.goal = goal

    def evaluate(self, action) -> float:
        return 1.0 if action else 0.0


class EasyRefundTask(CustomerSupportTask):
    def evaluate(self, action) -> float:
        if action == "refund":
            return 1.0
        elif  action == "apologize":
            return 0.5
        else:
         return 0.0


easy_task = EasyRefundTask(
    initial_state={
        "user_query": "I want refund for my order",
        "sentiment": "neutral",
        "issue_type": "refund_request",
        "order_status": "delivered",
        "attempts": 0,
    },
    goal="Agent should issue a refund.",
)


medium_task = CustomerSupportTask(
    initial_state={
        "user_query": "My package is late and I need an update.",
        "sentiment": "frustrated",
        "issue_type": "delivery_delay",
        "order_status": "in_transit",
        "attempts": 1,
    },
    goal="Provide a helpful response for a delayed delivery issue.",
)


hard_task = CustomerSupportTask(
    initial_state={
        "user_query": "I was charged twice and my replacement still has not arrived.",
        "sentiment": "angry",
        "issue_type": "billing_and_replacement",
        "order_status": "processing",
        "attempts": 2,
    },
    goal="Handle a multi-issue support case with a clear next step.",
)

#test code
# if __name__ == "__main__":
    task = easy_task

    print("Goal:", task.goal)

    print("Correct:", task.evaluate("refund"))
    print("Partial:", task.evaluate("apologize"))
    print("Wrong:", task.evaluate("track_order"))