from typing import Mapping, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState


class StrikerStrategy:

    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []  # Stores per-step rewards

        self.actor = Actor(state_dim=12, action_dim=5)
        self.critic = Critic(state_dim=12)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def reset(self, seed=None) -> None:
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()

    def record_reward(self, reward: float) -> None:
        """Stores the intermediate step reward returned by env.step()."""
        self.rewards.append(reward)

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}

        state_t = self.observe(state, config)
        if state_t is None:
            return {}

        dist = self.actor(state_t)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1).squeeze()
        val = self.critic(state_t).squeeze()

        self.log_probs.append(log_prob)
        self.values.append(val)

        controlled_id = config.controlled_robot_ids[0]
        return self.create_intents(action, controlled_id)

    def finish(self, final_reward: float = 0.0) -> None:
        """Trains the model using accumulated step-by-step trajectory rewards."""
        if len(self.values) <= 1 or len(self.rewards) == 0:
            self.reset()
            return

        values_t = torch.stack(self.values).squeeze()
        log_probs_t = torch.stack(self.log_probs).squeeze()

        # Compute GAE across the list of per-step rewards
        returns, advantages = self.compute_gae_and_returns(self.rewards, values_t)

        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = (returns.detach() - values_t).pow(2).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.reset()

    def observe(self, state: WorldState, config: TeamConfig) -> Optional[Tensor]:
        """Converts world state into relative coordinates for translation invariance."""
        striker = state.robots_blue.get(config.controlled_robot_ids[0])
        goalkeeper = state.robots_yellow.get(0)
        ball = state.ball

        if striker is None or goalkeeper is None or ball is None:
            return None

        # Relative vector to ball
        dx_ball = ball.x_m - striker.x_m
        dy_ball = ball.y_m - striker.y_m
        dist_ball = float(np.hypot(dx_ball, dy_ball))

        # Angle to ball relative to robot heading in [-pi, pi]
        angle_to_ball = float(np.arctan2(dy_ball, dx_ball) - striker.heading_rad)
        angle_to_ball = float(np.arctan2(np.sin(angle_to_ball), np.cos(angle_to_ball)))

        # Relative vector to goalkeeper
        dx_gk = goalkeeper.x_m - striker.x_m
        dy_gk = goalkeeper.y_m - striker.y_m

        return torch.tensor([
            dx_ball,
            dy_ball,
            dist_ball,
            angle_to_ball,
            striker.heading_rad,
            float(striker.infrared or 0),
            dx_gk,
            dy_gk,
            goalkeeper.heading_rad,
            ball.vx_mps,
            ball.vy_mps,
            float(np.hypot(ball.vx_mps, ball.vy_mps)),
        ], dtype=torch.float32).unsqueeze(0)

    def create_intents(self, action_t: torch.Tensor, robot_id: int) -> Mapping[int, RobotIntent]:
        action = action_t.detach().numpy().flatten()
        return {
            robot_id: RobotIntent(
                vx_mps=float(np.clip(action[0], -2.0, 2.0)),
                vy_mps=float(np.clip(action[1], -2.0, 2.0)),
                angular_velocity_rad_s=float(np.clip(action[2], -3.0, 3.0)),
                kick_speed_mps=float(np.clip(action[3], 0.0, 5.0)),
                dribbler=bool(action[4] > 0.0),
            )
        }

    def compute_gae_and_returns(
        self, rewards: list, values: Tensor, gamma=0.99, gae_lambda=0.95
    ) -> tuple[Tensor, Tensor]:
        advantages = []
        gae = 0.0

        values_detached = values.detach()

        for t in reversed(range(len(values_detached))):
            v_next = 0.0 if t == len(values_detached) - 1 else values_detached[t + 1].item()
            done_mask = 0.0 if (t == len(values_detached) - 1) else 1.0

            td_error = rewards[t] + gamma * v_next * done_mask - values_detached[t].item()
            gae = td_error + gae_lambda * gamma * done_mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        return returns, advantages


class Actor(nn.Module):
    def __init__(self, state_dim=12, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, action_dim)
        # Reduced initial exploration noise (e^-0.5 ~ 0.6) to prevent directional chatter
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, state: Tensor) -> Normal:
        mean = self.mean(self.net(state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)