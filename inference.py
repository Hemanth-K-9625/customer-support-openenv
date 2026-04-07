import os
from openai import OpenAI
from customer_support_env.server.customer_support_env_environment import (
    CustomerSupportEnvironment,
)

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

VALID_ACTIONS = ["respond_to_user", "check_order_status", "issue_refund"]


def get_action_from_llm(obs, history):
    prompt = f"""
You are a customer support agent.

Based on the customer message, choose the BEST action from:
{VALID_ACTIONS}

Rules:
- Only return ONE action
- No explanation
- Output must be exactly one of the actions

Customer message:
{obs}

Previous actions:
{history}

Answer:
"""

    try:
        response = requests.post(API_URL, headers=headers, json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 20
            }
        })

        data = response.json()

        # Extract text safely
        if isinstance(data, list) and "generated_text" in data[0]:
            output = data[0]["generated_text"].lower()

            # Match valid action
            for action in VALID_ACTIONS:
                if action in output:
                    return action

        # fallback
        return smart_fallback(obs)

    except Exception as e:
        print("LLM Error:", e)
        return smart_fallback(obs)

def smart_fallback(obs):
    obs = obs.lower()

    if "refund" in obs:
        return "issue_refund"
    elif "late" in obs or "delay" in obs:
        return "check_order_status"
    else:
        return "respond_to_user"
    
def parse_action(raw_output: str) -> str:
    for action in VALID_ACTIONS:
        if action in raw_output:
            return action

    if any(word in raw_output for word in ["refund", "money back", "return"]):
        return "refund"
    elif any(word in raw_output for word in ["sorry", "apologize", "apology"]):
        return "apologize"
    elif any(word in raw_output for word in ["status", "update", "track", "where"]):
        return "provide_status_update"
    elif any(word in raw_output for word in ["escalate", "human", "manager"]):
        return "escalate_to_human"
    elif any(word in raw_output for word in ["discount", "compensation", "coupon"]):
        return "give_discount"
    elif any(word in raw_output for word in ["acknowledge", "understand"]):
        return "acknowledge_issues"
    elif any(word in raw_output for word in ["close", "resolve", "done"]):
        return "close_case"
    
    return "apologize"

env = CustomerSupportEnvironment()
task = "medium"

# -----------------------------
# 🔹 START LOG (STRICT)
# -----------------------------
print(f"[START] task={task} env=CustomerSupportEnvironment model={MODEL_NAME}")

obs = env.reset(task=task)

total_reward = 0.0
step = 0
done = False

# -----------------------------
# 🔹 LOOP
# -----------------------------
while not done and step < 10:
    step += 1

    action = get_action_from_llm(obs, env.action_history)

    obs, reward, done, info = env.step(action)

    total_reward += reward

    # -----------------------------
    # 🔹 STEP LOG (STRICT)
    # -----------------------------
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}"
    )

# -----------------------------
# 🔹 END LOG (STRICT)
# -----------------------------
print(
    f"[END] success={str(done).lower()} steps={step} rewards={total_reward:.2f}"
)