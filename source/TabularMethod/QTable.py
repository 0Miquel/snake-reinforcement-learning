def qLearning(agent, env, qTable, snakeHead, score):
    learning_rate = 0.1  # usually used 0.1, it states the importance of the new information gained
    discount_factor = 0.9 # if close to 1 more importance to long term rewards, else only considers low term reward

    actualState = agent.get_state(env) # getting the actual state from the environment
    actualState = actualState.astype(str)
    actualState = ''.join(actualState) # convert actualState to string

    '''if actualState in qTable:
        actualRewards = qTable[actualState] #array of three qvalues, each for a diferent actin
    else:
        qTable[actualState] = [0,0,0]'''
    if actualState not in qTable: # add new state to the dictionary if it wasn't there
        qTable[actualState] = [0, 0, 0]

    action = agent.get_action(actualState, env, qTable)  # get action for the actual state

    observation, reward, done, info = env.step(action)
    env.render()

    if reward is 100:
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

    reward = agent.getReward2(env, action, done, snakeHead, appleEaten)

    # we calculate the new value of the qTable, using the reward from the previous action and updating it on the table

    qTable[lastState][action] = qTable[lastState][action] + \
    learning_rate * (reward + discount_factor*max(qTable[actualState]) - qTable[lastState][action])

    return done, score
