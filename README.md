# Course LLM Recommendation
这是一个采用集成模型作为用户模拟器给出反馈，RL作为整体的推荐智能体（单Agent）
1.初始化：
加载用户历史交互数据和候选项目特征。
使用LLM对候选项目进行分析，生成类别信息和关键词。
2.推荐决策：（DQN强化学习的action部分，根据状态给出下一步的动作）
推荐系统根据当前状态（用户历史行为和上下文信息）选择一个候选项目推荐给用户。
3.用户反馈模拟：（LLM+逻辑模型和统计模型作为奖励信号的传递，用于指导DQN的action）
用户模拟器根据候选项目的特征和用户历史行为，使用集成模型（LLM+逻辑＋统计）推断用户对项目的反馈（如“喜欢”或“不喜欢”）。<br />

## How to use my code?
你可以直接在 Course_DQN_main.py 里运行代码，记得把里面的绝对路径改为相对路径<br />

### Trainning environments
训练环境为自己定义的课程推荐环境，可以在environment.py里面进行查看<br />
### User simulator
用户模拟器可以在simulator.py进行查看，具体包含了两个逻辑模型以及一个统计模型，用于对DQN推荐的结果进行打分，作为奖励<br />
### How to see the training results?
你可以直接运行render.py中来得到运行结果，目前的结果只是训练其中一名学生的课程推荐结果<br />


