import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.actor  = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)

    def forward(self, x):
        f   = self.shared(x)
        mu  = self.actor(f)
        std = self.log_std.exp().expand_as(mu)
        val = self.critic(f).squeeze(-1)
        return mu, std, val

class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int,
                 lr=1e-4, gamma=0.99, clip=0.15, epochs=5, n_steps=1024):
        self.gamma = gamma
        self.clip = clip
        self.epochs  = epochs
        self.n_steps = n_steps
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.net     = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt     = optim.Adam(self.net.parameters(), lr=lr)

    def _act(self, obs):
        obs_t        = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mu, std, val = self.net(obs_t)
        dist         = torch.distributions.Normal(mu, std)
        a            = dist.sample()
        lp           = dist.log_prob(a).sum(-1)
        return a.squeeze(0).cpu().numpy(), lp.item(), val.item()        

    def collect_data(self, env, steps=500):
        obs_l, act_l, rew_l, val_l, lp_l, done_l = [], [], [], [], [], []

        obs, _ = env.reset()

        for _ in range(self.n_steps):
            a, lp, v = self._act(obs)
            nobs, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); rew_l.append(r)
            val_l.append(v);   lp_l.append(lp); done_l.append(float(term or trunc))
            obs = nobs if not (term or trunc) else env.reset()[0]
 
        advantages, returns = [], []
        gae = 0.0
        vals_ext = val_l + [0.0]
        for t in reversed(range(len(rew_l))):
            delta = rew_l[t] + self.gamma * vals_ext[t+1] * (1 - done_l[t]) - vals_ext[t]
            gae   = delta + self.gamma * 0.95 * (1 - done_l[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + vals_ext[t])
 
        f = lambda x: torch.FloatTensor(np.array(x)).to(self.device)
        return f(obs_l), f(act_l), f(lp_l), f(advantages), f(returns)
    
    def update(self, obs, acts, old_lp, advantages, returns):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
        for _ in range(self.epochs):
            mu, std, val = self.net(obs)
            dist  = torch.distributions.Normal(mu, std)
            lp    = dist.log_prob(acts).sum(-1)
            ratio = (lp - old_lp).exp()
 
            pl   = -torch.min(ratio * advantages,
                              ratio.clamp(1 - self.clip, 1 + self.clip) * advantages).mean()
            vl   = (val - returns).pow(2).mean()
            loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
 
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.opt.step()
 
        return {"policy_loss": pl.item(), "value_loss": vl.item()}
    
    def evaluate_on_env(self, env):
        values, weights = [1.0], []
        obs, _ = env.reset()
        env._t = env._t_start = 50
        env._weights = np.ones(env.n, dtype=np.float32) / env.n
        env._value   = 1.0
        env._returns = []
        obs = env._obs()
 
        done = False
        self.net.eval()
        with torch.no_grad():
            while not done:
                obs_t        = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                mu, _, _     = self.net(obs_t)
                a            = mu.squeeze(0).cpu().numpy()   
                obs, _, term, trunc, info = env.step(a)
                values.append(info["value"])
                weights.append(info["weights"])
                done = term or trunc
        return np.array(values), np.array(weights)
 
    def train(self, env_train, env_val, epochs=200, patience=5, save_path="models/ppo"):
        os.makedirs(save_path, exist_ok=True)
        best_val = -np.inf
        history  = []
        no_improve = 0
 
        for i in range(1, epochs + 1):
            self.net.train()
            obs, acts, lp, adv, ret = self.collect_data(env_train)
            metrics = self.update(obs, acts, lp, adv, ret)
 
            if i % 10 == 0:
                val_values, _ = self.evaluate_on_env(env_val)
                val_return    = val_values[-1] - 1.0
                history.append({"epoch": i, "val_return": val_return, **metrics})
                
 
                if val_return > best_val:
                    best_val   = val_return
                    no_improve = 0
                    torch.save(self.net.state_dict(), f"{save_path}/best.pt")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\n  Early stopping en epoch {i} | mejor val={best_val:+.4f}")
                        break
                print(f"Epoch {i:3d}/{epochs} | val_return={val_return:+.4f} "
                      f"| best={best_val:+.4f} "
                      f"| pl={metrics['policy_loss']:.4f} vl={metrics['value_loss']:.4f}")
                
 
                
 
        self.load(f"{save_path}/best.pt")
        print(f"\nEntrenamiento completado. Mejor val_return={best_val:+.4f}")
        return history
 
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()
 

    

