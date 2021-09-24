from gym.envs.registration import register

register(
    id="Cartpole-v0",
    entry_point="norse_rl.cartpole:CartpoleEnv",
)
register(
    id="Gridworld-v0",
    entry_point="norse_rl.gridworld:GridworldEnv",
)
