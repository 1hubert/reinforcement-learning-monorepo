import random
import time

class GenshinAgent:
    def __init__(self):
        self.ability_a_cooldown = 0
        self.ability_b_cooldown = 0

    def use_ability_a(self):
        if time.perf_counter() < self.ability_a_cooldown:
            return "Ability A on cooldown"
        print("Used Ability A")
        self.ability_a_cooldown = time.perf_counter() + 12

    def use_ability_b(self):
        if time.perf_counter() < self.ability_b_cooldown:
            return "Ability B on cooldown"
        print("Used Ability B")
        self.ability_b_cooldown = time.perf_counter() + 10

    def take_action(self):
        action = random.choice(["a", "b"])
        if action == "a":
            return self.use_ability_a()
        elif action == "b":
            return self.use_ability_b()

agent = GenshinAgent()
while True:
    action_result = agent.take_action()
    if action_result:
        print(action_result)
    # time.sleep(0.5) # sleep so we can see the outputs
