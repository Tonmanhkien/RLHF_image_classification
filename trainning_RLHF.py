import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.distributions import Categorical
from base_model import SimpleCNN

class FeedbackDataset(Dataset):
    def __init__(self, feedback_json, cifar_dataset):
        with open(feedback_json) as f:
            self.fb = json.load(f)
        self.base = cifar_dataset    # your CIFAR10 testset with transform applied

    def __len__(self):
        return len(self.fb)

    def __getitem__(self, i):
        idx, corr = self.fb[i]["index"], self.fb[i]["correct"]
        image, _ = self.base[idx]
        reward = torch.tensor([1.0 if corr else -1.0], dtype=torch.float32)
        return image, reward
    
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = SimpleCNN()

        self.features.fc2 = nn.Identity()

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)      # → (batch_size, 512)
        reward = self.head(x)     # → (batch_size, 1), in [0,1]
        return reward
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

feedback_json = 'feedback.json'
dataset = FeedbackDataset(feedback_json, cifar_test)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

reward_model = RewardModel().to(device)
optimizer = optim.Adam(reward_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train_reward_model(num_epochs=10):
    reward_model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = reward_model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

    torch.save(reward_model.state_dict(), 'reward_model.pth')
    print("Saved reward_model.pth")

train_reward_model()

class PolicyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=None)      
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return F.softmax(self.backbone(x), dim=1)
    
class PPO:
    def __init__(self, policy, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.policy     = policy
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma      = gamma
        self.eps_clip   = eps_clip

    def update(self, states, actions, rewards, old_probs):
        # stack everything
        states      = torch.cat(states,    dim=0).to(device)
        actions     = torch.cat(actions,   dim=0).to(device)
        rewards     = torch.cat(rewards,   dim=0).to(device)
        old_probs   = torch.cat(old_probs, dim=0).to(device)

        # compute discounted returns
        returns = []
        R = 0
        for r in rewards.flip(0):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=device)

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # get new action probs
        probs = self.policy(states)
        dist  = Categorical(probs)
        new_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ratio for PPO
        ratio = new_probs / old_probs
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * returns
        loss  = -torch.min(surr1, surr2).mean()

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
def to_onehot(idx, K):
    oh = torch.zeros(K, device=device)
    oh[idx] = 1.0
    return oh

policy_model = PolicyModel(num_classes=10).to(device)
ppo_agent    = PPO(policy_model)
reward_model.load_state_dict(torch.load('reward_model.pth'))
reward_model.eval()

def train_rlhf_ppo(num_epochs=5, batch_size=8):
    policy_model.train()
    for epoch in range(num_epochs):
        states, actions, rewards, old_probs = [], [], [], []
        epoch_loss = 0.0
        for step, (img, _) in enumerate(loader):
            img = img.to(device)

            # 1) policy selects action
            probs = policy_model(img)                         # [1,K]
            dist  = Categorical(probs)
            a     = dist.sample()                             # [1]
            prob  = probs.gather(1, a.unsqueeze(1)).squeeze(1)  # [1]

            # 2) reward_model scores (state,action)
            action_oh = to_onehot(a.item(), 10).unsqueeze(0)  # [1,K]
            r = reward_model(img, action_oh)                    # [1]

            # collect trajectory
            states.append(img)
            actions.append(a.unsqueeze(0))
            rewards.append(r.unsqueeze(0))
            old_probs.append(prob.unsqueeze(0))

            # once we have enough for a PPO update
            if len(states) >= batch_size:
                loss = ppo_agent.update(states, actions, rewards, old_probs)
                epoch_loss += loss
                states, actions, rewards, old_probs = [], [], [], []

        print(f'Epoch {epoch+1}/{num_epochs} — Avg PPO loss: {epoch_loss:.4f}')

    # save the final policy
    torch.save(policy_model.state_dict(), 'rlhf_policy.pth')
    print("Saved rlhf_policy.pth")

train_rlhf_ppo()