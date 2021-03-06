def qLearning(agent, env, qTable, snakeHead, score):
    actualState = agent.get_state(env) # getting the actual state from the environment
    actualState = actualState.astype(str)
    actualState = ''.join(actualState) # convert actualState to string

    if actualState not in qTable: # add new state to the dictionary if it wasn't there
        qTable[actualState] = [0, 0, 0]

    action = agent.get_action(actualState, env, qTable)  # get action for the actual state

    observation, reward, done, info = env.step(action)
    env.render()

    if reward > 0:
        appleEaten = True
        score = score + 1
    else:
        appleEaten = False

    lastState = actualState #last state is the state where we chose an action

    actualState = agent.get_state(env)  # getting the actual state from the environment
    actualState = actualState.astype(str)
    actualState = ''.join(actualState)  # convert actualState to string

    # actualState is the result from the action taken
    if actualState not in qTable: # add new state to the dictionary if it wasn't there
        qTable[actualState] = [0, 0, 0]

    reward = agent.getReward(env, action, done, snakeHead, appleEaten)

    # we calculate the new value of the qTable, using the reward from the previous action and updating it on the table

    qTable[lastState][action] = qTable[lastState][action] + \
    agent.learningRate * (reward + agent.gamma*max(qTable[actualState]) - qTable[lastState][action])

    return done, score

def sarsa(action, agent, env, qTable, snakeHead, score, actualState):
    if action is None:
        actualState = agent.get_state(env) # getting the actual state from the environment
        actualState = actualState.astype(str)
        actualState = ''.join(actualState) # convert actualState to string

        if actualState not in qTable: # add new state to the dictionary if it wasn't there
            qTable[actualState] = [0, 0, 0]

        action = agent.get_action(actualState, env, qTable)  # get action for the actual state

    observation, reward, done, info = env.step(action)
    env.render()

    if reward > 0:
        appleEaten = True
        score = score + 1
    else:
        appleEaten = False

    lastState = actualState #last state is the state where we chose an action

    actualState = agent.get_state(env)  # getting the actual state from the environment
    actualState = actualState.astype(str)
    actualState = ''.join(actualState)  # convert actualState to string

    # actualState is the result from the action taken
    if actualState not in qTable: # add new state to the dictionary if it wasn't there
        qTable[actualState] = [0, 0, 0]

    reward = agent.getReward(env, action, done, snakeHead, appleEaten)

    newAction = agent.get_action(actualState, env, qTable)  # get action for the new state (after first action)

    # we calculate the new value of the qTable, using the reward from the previous action and updating it on the table

    qTable[lastState][action] = qTable[lastState][action] + \
        agent.learningRate * (reward + agent.gamma*qTable[actualState][newAction] - qTable[lastState][action])

    return done, score, newAction, actualState
