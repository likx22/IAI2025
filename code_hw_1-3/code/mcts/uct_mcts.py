from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        max_ucb = -float('inf')
        best_action = None
        total_visits = np.sum(node.child_N_visit)
    
        all_unvisited = (node.child_N_visit == 0).all()

        for action in range(node.n_action):
            if node.action_mask[action] == 0:
                continue

            if all_unvisited:
                best_action = np.argmax(node.action_mask)  # Choose the leftmost valid action
                break

            if node.child_N_visit[action] == 0:
                exploit = 0  # 未访问过的节点，避免选择
            else:
                exploit = node.child_V_total[action] / node.child_N_visit[action]  # Exploit: average value

            explore = np.sqrt(np.log(total_visits + 1) / (node.child_N_visit[action] + 1))  # Explore: exploration factor
            ucb = exploit + self.config.C * explore

            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action

        return best_action
        ########################

    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################
        while node is not None:
            node.child_N_visit[node.action] += 1
            node.child_V_total[node.action] += value

            if node.parent:  # Ensure parent exists
                action = node.action
                # Update parent's statistics
                node.parent.child_N_visit[action] += 1
                node.parent.child_V_total[action] += value
                value=-value
        
            node = node.parent
        ########################    
            
    
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!
        
        ########################
        # TODO: your code here #
        ########################
        env = node.env.fork()  # Fork the environment to avoid side effects
        done = False
        current_player = env.current_player

        while not done:
            valid_actions = np.where(env.action_mask == 1)[0]  # Get valid actions
            action = np.random.choice(valid_actions)  # Random action selection
            _, reward, done = env.step(action)  # Perform action and get result

        # Return the reward from the current player's perspective
        return 3*reward if env.current_player == current_player else -3*reward
        ########################
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        node = self.root
        while not node.done:
            if len(node.children) < node.n_action:  # If node is not fully expanded
                for action in range(node.n_action):
                    if node.action_mask[action] == 1 and not node.has_child(action):
                        return node.add_child(action)  # Expand a new child node
            # Select the best child based on UCT
            action = self.uct_action_select(node)
            node = node.get_child(action)

        return node
        ########################
    
    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
        ########################
        # TODO: your code here #
        ########################
        if node is None:
            node = self.root
        policy = np.zeros(node.n_action)

        for action in range(node.n_action):
            if node.action_mask[action] == 1:
                policy[action] = node.child_N_visit[action]

        # Normalize the policy
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # If all visit counts are zero, use a uniform distribution over valid actions
            valid_actions = np.where(node.action_mask == 1)[0]
            policy[valid_actions] = 1 / len(valid_actions)

        return policy
        ########################

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended) 
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        ########################
        # TODO: your code here #
        ########################
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
        
            if leaf.done:
                value = leaf.reward
            else:
                value_total = 0
                for _ in range(self.config.n_rollout):
                    value_total += self.rollout(leaf)
                value = value_total / self.config.n_rollout
        
            self.backup(leaf, value)
    
        return self.get_policy(self.root)
        ########################