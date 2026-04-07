---
title: Customer Support OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---

# Customer Support OpenEnv

This is a customer support simulation environment built using OpenEnv.

## Description
This project simulates customer support tasks (easy, medium, hard) for training and evaluation.

## How to Run

### Locally
pip install -r requirements.txt  
python inference.py  

### Docker
docker build -t cs-env .  
docker run cs-env  

## Tasks
- Easy: Simple refund request  
- Medium: Delayed order  
- Hard: Multiple issues (wrong item + delay)