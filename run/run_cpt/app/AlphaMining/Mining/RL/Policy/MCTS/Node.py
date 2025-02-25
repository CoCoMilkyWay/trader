# Node class for MCTS
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.action = action
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            next_state = self.state.copy()
            next_state[self.parent.counter] = action
            self.children[action] = MCTSNode(
                state=next_state, parent=self, action=action, prior_prob=prior
            )

    def update(self, value):
        self.visit_count += 1
        self.total_value += value

    def ucb_score(self, c_puct):
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.total_value / self.visit_count
        u_value = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + u_value