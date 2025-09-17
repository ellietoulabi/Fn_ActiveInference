from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

env = SingleAgentRedBlueButtonEnv()

state = env.reset()

print(state)

action = env.action_space.sample()

state, reward, done, info = env.step(action)