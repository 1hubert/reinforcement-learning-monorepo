class QL:
    def __init__(self):
        self.a = 1
        self.b = 2

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = [0, 0, 0]

ql = QL()
ql.reset_qtable()
print(ql.qtable)
