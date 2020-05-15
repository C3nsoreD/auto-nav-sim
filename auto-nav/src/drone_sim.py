""" A 2D drone simulator for testing autonomous flight """

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
    
    def set_drone_state(self, state):
        self.drone_state = [state[0], state[1], state[2], state[3]]

    def reset(self):
        pass

    def step(self):
        pass

    
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