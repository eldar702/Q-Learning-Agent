# Q-Learning-Agent

Consists of 2 agents:
1. QLearningAgent: The above agent is a learning agent. That is - he is the agent who creates and learns the policy. The result of his run is the policy file.
    Example of the policy file:
    ![image](https://user-images.githubusercontent.com/72104254/169683171-9277d392-a8da-4c89-925a-96d91828c789.png)
    
    And of course, the update process is done using the Bellman equation:
    Q(s,a) := (1 - alpha) * Q(s,a) + alpha * [R(s,a) + gamma * max Q(s',a')]
    
2. QExecutorAgent: The above agent is the executor agent. That is - he is the agent who uses the policy file that the learner agent has learned. He does not change and update it, but only uses it.
