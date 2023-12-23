abilities = ['fireball', 'heal', 'move left']
def ability_on_cooldown(ability):
    if ability == 'fireball':
        return True
    return False
ready_abilities = [a for a in abilities if not ability_on_cooldown(a)]
print(ready_abilities)
