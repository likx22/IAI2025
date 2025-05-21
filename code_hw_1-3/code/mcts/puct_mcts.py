from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np
import random


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
       # select the best action based on PUCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        max_ucb = -float('inf')
        best_action = None
        total_visits = node.child_N_visit.sum()
        
        pucb = node.child_V_total / (node.child_N_visit + 1) \
            + node.child_priors * self.config.C \
                * np.sqrt(0.618 * np.log(total_visits) / (1 + node.child_N_visit))
    
        pucb[node.action_mask == 0] = -INF
        
        return np.argmax(pucb)
        ########################

    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        
        ########################
        # TODO: your code here #
        ########################
        while node.parent is not None:
            action = node.action  
            parent = node.parent
        
            parent.child_N_visit[action] += 1
            parent.child_V_total[action] += value
            
            value = -value                             
            node = node.parent
        ########################  
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        node = self.root
        while not node.done:
            # Check if all valid actions have been explored
            unexplored = [act for act in np.where(node.action_mask == 1)[0]if not node.has_child(act)]
        
            if unexplored:
                # Choose an unexplored action
                action = random.choice(unexplored)
                new_node = node.add_child(action)

                if not new_node.done:
                    canonical_obs = new_node.env.compute_canonical_form_obs(new_node.env.observation, new_node.env.current_player)
                    policy, _ = self.model.predict(canonical_obs)
                    new_node.set_prior(policy)
            
                return new_node
            else:
                # All actions have been explored, choose the best child according to PUCT
                action = self.puct_action_select(node)
                node = node.get_child(action)
    
        return node
        ########################
    
    def get_policy(self, node:MCTSNode = None):
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
    
        # Normalize
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
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                value += leaf.reward
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                # NOTE: you should compute the policy and value 
                #       using the value&policy model!
                obs = leaf.env.observation
                player=leaf.env.current_player
                canonical_obs = leaf.env.compute_canonical_form_obs(obs, player)
                child_prior, value = self.model.predict(canonical_obs)
                leaf.set_prior(child_prior)
                ########################
            self.backup(leaf, value)
        self.root.set_prior(self.get_policy(self.root))
        return self.get_policy(self.root)