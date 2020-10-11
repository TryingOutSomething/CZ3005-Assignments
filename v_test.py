from pprint import pprint

from agents.value_iteration_agent import ValueIterationAgent
from environment import TreasureCube

env = TreasureCube(max_step=5000)
agent = ValueIterationAgent(env.dim)

agent.train()

pprint(agent.v_table)
