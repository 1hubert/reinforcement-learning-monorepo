import random
import time

class GenshinAgent:

    def __init__(self):
        self.actions = [self.fireball, self.heal]
        self.last_used = {
            self.fireball: 0,
            self.heal: 0
        }

    def fireball(self):
        print("Used Fireball")
        self.last_used[self.fireball] = time.time() + 5

    def heal(self):
        print("Used Heal")
        self.last_used[self.heal] = time.time() + 10

    def action_on_cooldown(self, action):
        last_used = self.last_used.get(action, 0)
        return  time.time() < last_used

    def get_ready_action(self):
        ready_actions = [a for a in self.actions if not self.action_on_cooldown(a)]
        if len(ready_actions) > 0:
            return random.choice(ready_actions)

    def take_action(self):
        action = self.get_ready_action()
        if action:
            action()
        else:
            print("All abilities on cooldown")

agent = GenshinAgent()

while True:
    agent.take_action()
    time.sleep(0.5)
