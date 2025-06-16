import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from PIL import Image
import numpy

HEIGHT = 160
WIDTH = 210
DOWN = 110
CROP = 84

CROP_LEFT = (DOWN - CROP) / 2

GAMMA = 0.9

gym.register_envs(ale_py)

env = gym.make("ALE/Pong", render_mode="human")

observation, info = env.reset(seed=42)

def process(observation):
    img = Image.fromarray(observation)
    grayscale = img.convert('L')
    down = grayscale.resize((DOWN,CROP), resample=Image.BILINEAR)
    cropped = down.crop((CROP_LEFT,0, CROP_LEFT + CROP, CROP))
    return cropped


class Q_Function(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=6)
        
    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.lin(x)
        x = self.out(x)
        return x
        



q = Q_Function(16, 8, 4)


for _ in range(10):
    history = []
    observation, info = env.reset(seed=42)
    for _ in range(4):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        history.append(process(observation))
    n_hist = numpy.asanyarray(history, dtype=numpy.float32)
    input = torch.from_numpy(n_hist)
    
    output = q.forward(input)
    loss = (reward + GAMMA(output.max(keepdim=True)) - output[action])
    print(n_hist.shape, output)
    if terminated or truncated:
        observation, info = env.reset()
    



env.close()

