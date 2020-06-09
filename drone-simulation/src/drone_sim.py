""" A 2D drone simulator for testing autonomous flight """
import math 
import random.random

class DroneState(object):
    def __init__(self):
        """
        Class constructor that initializes the drone state varibales.
        These give us information about the drones position in space and time.
        """
        self.x = 0.0
        self.y = 0.0
        self.rot_x = 0.0 
        self.rot_y = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.drone_state = [self.x, self.y, self.x_dot, self.y_dot]    
        self.done = False 
        self.is_reset = False 
        # constants
        self.Q = dict() # The action value function results / table 
        self.epsilon_constant = 1.0

    # def set_drone_state(self, state):
    #     self.drone_state = [state[0], state[1], state[2], state[3]]

    def build_state(self, state):
        # Returns the current state as a string value based on the input arg state: x_y
        state = str(int(round(state[0])))+'_'+str(int(round(state[1])))

        return state

    def get_maxQ(self, state):
        max_Q = -10000000 
        
        for action in self.Q[state]:
            if self.Q[state][action] > max_Q:
                max_Q = self.Q[state][action]

        return max_Q
    
    def create_Q(self, state):
        """
        Creates the action value function and sets the current state to 0,0
        """
        if state is not self.Q.keys():
            self.Q[state] = self.Q.get(state, {'0':0.0, '1':0.0, '2':0.0, '3':0.0})

        return
    
    def choose_action(self, state):
        valid_actions = ['0','1','2','3']
        """ Choice function that implements a greedy algorithm 
        """
        if random.random() < self.epsilon_constant:
            action = random.choice(valid_actions)
            print('Random: ', action)
        else:
            action = []
            max_Q = self.get_maxQ(state)
            for action in self.Q[state]:
                if self.Q[state][action] == float(max_Q):
                    action.append(action)
            action = random.choice(action)
            print('Greddy: ', action)

        return action
    
    def learn(self, state, action, reward, next_state):
        maxQ_next_state = drone.get_maxQ(next_state)
        # self.Q[state][action] = self.Q[state][action] + self.alpha*(reward - self.Q[state][action])
        self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + self.alpha*(self.gamma*(reward + maxQ_next_state))



drone = DroneState()

def get_reward(state):
    reward = 0.0
    """ Reward function that is based on the drones position in space 
    Positive reward given when the drones movement is within 0.05 in both x,y 
    Negative reward given when the drone is out of the boundaries of the 
    """
    if abs(state[0] - 5.0) <= 0.05 and abs(state[1] - 5.0) <= 0.05:
        # Positive reward
        reward = 1.0
    elif (state[0] < -0.1 or state[0] > 5.1 or state[1] < -0.1 or state[1] > 5.1 or reward == 1):
        # Negative reward
        reward = -100.0
    else: 
        reward = -math.sqrt((5.0-state[0])**2 + (5.0-state[0])**2)
    
    return reward

def take_off():
    """ Defines the take off position and state. The could be many states
    
    Direct line of sight to the target 

    Indirect or skew line of sight (positioned at an angle from the target)
    """
    pass

def main():
    pass 
