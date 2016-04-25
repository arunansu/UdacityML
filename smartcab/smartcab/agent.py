import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.last_reward = 0
        self.last_action = None
        self.last_state = None
        self.state = None
        self.total_reward = 0
        self.deadline = self.env.get_deadline(self)
        self.q = {}
        self.alpha = 0.9
        self.gamma = 0.3
        self.epsilon = 0.1
        self.actions = ['forward', 'left', 'right', None]

    def get_max_q(self, state):
        maxQ = 0.0
        for action in self.actions:
            if(self.q.get((state, action)) > maxQ):
                maxQ = self.q.get((state, action), 0.0)
        return maxQ

    def get_best_action(self, state):
        best_action = random.choice(self.actions)
        maxQ = self.get_max_q(state)
        if(random.random() > self.epsilon):
            for action in self.actions:
                if(self.q.get((state, action), 0.0) > maxQ):
                    best_action = action
                if(self.q.get((state, action), 0.0) == maxQ):
                    if(random.random() > 0.5):
                        best_action = action
        return best_action

    def tuple_state(self, state):
        State = namedtuple("State", ["light", "oncoming", "left", "next_waypoint"])
        return State(light = state['light'], oncoming = state['oncoming'], left = state['left'], next_waypoint = self.planner.next_waypoint()) 

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.last_reward = 0
        self.last_action = None
        self.last_state = None
        self.state = None
        self.total_reward = 0

    def update_q(self, state, action, next_state, reward):
        if((state, action) not in self.q):
            self.q[(state, action)] = 0.0
        else:
            self.q[(state, action)] = self.q[(state, action)] + self.alpha * (reward + self.gamma * self.get_max_q(next_state) - self.q[(state, action)])       

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        self.state = self.tuple_state(inputs)

        # TODO: Select action according to your policy
        action = self.get_best_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if(self.last_reward != None):
            self.update_q(self.last_state, self.last_action, self.state, self.last_reward)
        
        self.last_action = action
        self.last_state = self.state
        self.last_reward = reward
        self.total_reward += reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
