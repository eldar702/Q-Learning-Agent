import random
import sys
import numpy as np
from tabulate import tabulate
import os.path
##############################            Imports & Globals              #################################
from pddlsim.executors.executor import Executor
from pddlsim.local_simulator import LocalSimulator

# read files:
input_flag = sys.argv[1]
domain_path = sys.argv[2]
problem_path = sys.argv[3]
policy_file_path = sys.argv[4]

# GLOBAL's
LAST_STATE = None
LAST_ACTION = None

#########################################################################################################
###########################            BehaviorBaseAgent Class              #############################
#########################################################################################################
class QLearningAgent(Executor):
    ##########################             Init Functions               #################################
    def __init__(self):
        super(QLearningAgent, self).__init__()
        self.epsilon = 1
        self.Q_table = None
        self.actions_list, self.states_list, self.actions_idx, self.states_idx = [], [], {}, {}

    def initialize(self, services):
        self.services = services
        self.initialize_Q_table()


    def next_action(self):
        global LAST_ACTION, LAST_STATE
        chosen_action = None
        self.update_Q_table()
        self.write_Q_table()
        r = np.random.uniform(0, 1)
        self.change_epsilon()
        if self.services.goal_tracking.reached_all_goals():
            return None

        valid_actions = self.services.valid_actions.get()
        if len(valid_actions) == 0:
            return None

        elif len(valid_actions) == 1:
            chosen_action = self.services.valid_actions.get()[0]
        # explore
        elif r < self.epsilon:
            chosen_action = random.choice(valid_actions)
        # exploit
        elif r >= self.epsilon:
            chosen_action = self.choose_best_action(valid_actions)

        LAST_ACTION = chosen_action.split()[0].split('(')[1]
        LAST_STATE = self.get_agent_location()
        return chosen_action

    #######################             Q - TABLE  Methods               ################################
    def initialize_Q_table(self):

        self.actions_list = sorted([state for state in self.services.parser.actions])
        self.states_list = sorted([action for action in self.services.parser.objects if "person" not in action and "food" not in action])

        zero_matrix = np.zeros((len(self.states_list), len(self.actions_list)))
        self.Q_table = np.vstack([self.actions_list, zero_matrix])
        self.states_list.insert(0, "State/Action")
        self.Q_table = np.append([[state] for state in self.states_list], self.Q_table, axis=1)

        self.actions_idx = {k: v + 1 for v, k in enumerate(self.actions_list)}
        self.states_idx = {k: v for v, k in enumerate(self.states_list)}

       # table = tabulate(Q_table_val, self.temp_actions, tablefmt="fancy_grid")
       # print(table)

    def write_Q_table(self):
        f = open("POLICYFILE", "w")
        np.savetxt(f, self.Q_table, fmt="%10s")

    def read_Q_table(self):
        self.Q_table = np.loadtxt('POLICYFILE')

        self.actions_list = list(self.Q_table[0, 1:])
        self.states_list = self.Q_table[:, 0]
        self.states_list = [state for state in self.states_list]

        self.actions_idx = {k: v + 1 for v, k in enumerate(self.actions_list)}
        self.states_idx = {k: v for v, k in enumerate(self.states_list)}


    def update_Q_table(self):
        global LAST_ACTION, LAST_STATE
        if LAST_STATE is None or LAST_ACTION is None:
            return
        reward = self.get_reward(LAST_ACTION)

        #update the table
        self.Q_table[self.states_idx[LAST_STATE]][self.actions_idx[LAST_ACTION]] = reward


    ####################             Q - LEARNING  Methods               #############################
    def choose_best_action(self, valid_Actions):

        state = self.get_agent_location()
        best_action = []
        best_action_value = float('-inf')
        for action in valid_Actions:
            action_value = self.Q_table[self.states_idx[state]][self.actions_idx[action]]
            if action_value == best_action_value:
                best_action.append(action)
            elif action_value > best_action_value:
                best_action = [action]
        return random.choice(best_action)

    def change_epsilon(self):
        if self.epsilon > 0.3:
            self.epsilon *= 0.95


    def get_reward(self, action):
        if "pick-food" in action:
            return 100
        else:
            # last action is a step
            return -1


    ##############################             PDDL  Methods               #################################
    def get_agent_location(self):
        state = self.services.perception.get_state()
        agent_place = list(state["at"])
        agent_place = agent_place[0][1]
        return agent_place


##############################             HELPER's  Methods               #################################
def there_is_policy_file():
    return os.path.exists('POLICYFILE')

class QExecutorAgent(QLearningAgent):
    ##########################             Init Functions               #################################
    def __init__(self):
        super(QLearningAgent, self).__init__()


    def initialize(self, services):
        self.services = services
        self.read_Q_table()


    def next_action(self):

        if self.services.goal_tracking.reached_all_goals():
            return None

        valid_actions = self.services.valid_actions.get()
        if len(valid_actions) == 0:
            return None

        elif len(valid_actions) == 1:
            chosen_action = self.services.valid_actions.get()[0]

        else:
            chosen_action = self.choose_best_action(valid_actions)

        return chosen_action


    def choose_best_action(self, valid_Actions):

        state = self.get_agent_location()
        best_action = []
        best_action_value = float('-inf')
        for action in valid_Actions:
            action_value = self.Q_table[self.states_idx[state]][self.actions_idx[action]]
            if action_value == best_action_value:
                best_action.append(action)
            elif action_value > best_action_value:
                best_action = [action]
        return random.choice(best_action)

if input_flag == "-L":
    print LocalSimulator().run(domain_path, problem_path, QLearningAgent())

elif input_flag == "-E":
    print LocalSimulator().run(domain_path, problem_path, QExecutorAgent())
