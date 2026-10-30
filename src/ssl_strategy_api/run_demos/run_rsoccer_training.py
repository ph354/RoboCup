import argparse
import csv
from importlib import import_module
from math import ceil
import os
import time
from typing import Any, Dict, Optional, Tuple

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from ssl_strategy_api import FieldGeometry, TeamColor, TeamConfig
from ssl_strategy_api.adapters.rsoccer import (
    to_rsoccer_commands,
    world_state_from_rsoccer_frame,
)
from ssl_strategy_api.helpers import BallSpawner

SIMULATION_STEP_SECONDS = 0.025
DEFAULT_EPISODE_DURATION_SECONDS = 5.0
BLUE_ROBOT_ID = 0
YELLOW_ROBOT_ID = 0
RIGHT_SIDE_MARGIN_M = 0.20
BALL_IDLE_SPEED_MPS = 0.06
DEFAULT_NUM_EPISODES = 10000


class BaseRewardEvaluator:
    """Evaluates per-step dense rewards and finish criteria."""

    def __init__(self):
        self.prev_distance: Optional[float] = None

    def calculate_reward_and_done(
        self, env: "TrainingTouchBallEnv"
    ) -> Tuple[float, bool, Dict[str, Any]]:
        state = env.state
        frame = env.frame
        ball = state.ball if state is not None else None
        blue_robot = frame.robots_blue.get(BLUE_ROBOT_ID) if frame else None

        if blue_robot is None or frame.ball is None:
            return 0.0, False, {"finish_reason": "missing_entities", "touched_ball": False}

        current_dist = float(
            ((blue_robot.x - frame.ball.x) ** 2 + (blue_robot.y - frame.ball.y) ** 2) ** 0.5
        )

        if self.prev_distance is None:
            self.prev_distance = current_dist

        # Distance closed in meters (positive if moving closer)
        distance_closed = self.prev_distance - current_dist

        # Dense Reward: 100x distance progress minus small step penalty
        reward = (distance_closed * 100.0) - 0.01

        respawn_reason = env.ball_spawner.respawn_when_goal_or_out_or_idle(
            ball,
            env.field,
            attack_direction_x=env.blue_config.attack_direction_x,
            idle_speed_mps=BALL_IDLE_SPEED_MPS,
        )

        blue_in_opponent_goal_area = env.ball_spawner.in_opponent_goal_area(
            env.field,
            blue_robot.x,
            blue_robot.y,
            attack_direction_x=env.blue_config.attack_direction_x,
        )

        ball_touched = bool(blue_robot.infrared)

        done = False
        finish_reason = "in_progress"

        if ball_touched:
            done = True
            reward += 500.0
            finish_reason = "touched_ball"
        elif blue_in_opponent_goal_area:
            done = True
            reward -= 50.0
            finish_reason = "goal_area_foul"
        elif respawn_reason is not None:
            done = True
            reward -= 20.0
            finish_reason = f"out_or_idle_{respawn_reason}"

        self.prev_distance = current_dist if not done else None

        info = {
            "finish_reason": finish_reason,
            "touched_ball": ball_touched,
        }
        return reward, done, info


class TrainingTouchBallEnv(SSLBaseEnv):
    """Headless environment for rapid training sessions."""

    def __init__(
        self,
        spawner_seed: Optional[int] = None,
        evaluator: Optional[BaseRewardEvaluator] = None,
    ):
        self.blue_config = TeamConfig(
            team=TeamColor.BLUE,
            controlled_robot_ids=(BLUE_ROBOT_ID,),
            attack_direction_x=1,
        )
        self.yellow_config = TeamConfig(
            team=TeamColor.YELLOW,
            controlled_robot_ids=(YELLOW_ROBOT_ID,),
            attack_direction_x=-1,
        )
        super().__init__(
            field_type=1,
            n_robots_blue=1,
            n_robots_yellow=1,
            time_step=SIMULATION_STEP_SECONDS,
            render_mode=None,
        )
        self.canonical_field = FieldGeometry(
            length_m=self.field.length,
            width_m=self.field.width,
            goal_width_m=self.field.goal_width,
        )
        self.state = None
        self.ball_spawner = BallSpawner(seed=spawner_seed)
        self.evaluator = evaluator or BaseRewardEvaluator()
        self.latest_info: Dict[str, Any] = {}

    def _get_commands(self, commands):
        return commands

    def _frame_to_observations(self):
        self.state = world_state_from_rsoccer_frame(
            self.frame,
            self.canonical_field,
            frame_number=self.steps,
        )
        return self.state

    def _calculate_reward_and_done(self):
        reward, done, info = self.evaluator.calculate_reward_and_done(self)
        self.latest_info = info
        return reward, done

    def _get_initial_positions_frame(self):
        frame = Frame()
        blue_spawn = self.ball_spawner.spawn_stationary_right_side(
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            right_side_margin_m=RIGHT_SIDE_MARGIN_M,
            enforce_outside_opponent_goal_area=True,
        )
        ball_spawn = self.ball_spawner.spawn_stationary_right_side(
            self.field,
            attack_direction_x=self.blue_config.attack_direction_x,
            right_side_margin_m=RIGHT_SIDE_MARGIN_M,
            enforce_outside_opponent_goal_area=True,
        )

        frame.ball = Ball(
            x=ball_spawn.x_m,
            y=ball_spawn.y_m,
            v_x=ball_spawn.vx_mps,
            v_y=ball_spawn.vy_mps,
        )

        frame.robots_blue[BLUE_ROBOT_ID] = Robot(
            yellow=False,
            id=BLUE_ROBOT_ID,
            x=blue_spawn.x_m,
            y=blue_spawn.y_m,
            theta=180.0,
        )

        half_length = self.field.length / 2
        half_width = self.field.width / 2
        frame.robots_yellow[YELLOW_ROBOT_ID] = Robot(
            yellow=True,
            id=YELLOW_ROBOT_ID,
            x=half_length - 0.10,
            y=-(half_width),
            theta=180.0,
        )
        return frame


def _load_strategy(path: str) -> Any:
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError("strategy must use module.path:ClassName format")
    strategy_class = getattr(import_module(module_name), class_name)
    return strategy_class()


def train(
    blue_strategy: Any,
    yellow_strategy: Any,
    num_episodes: int,
    max_duration_seconds: float = DEFAULT_EPISODE_DURATION_SECONDS,
    log_filename: str = "training_stats.csv",
    log_freq: int = 1,
    seed: Optional[int] = None,
) -> None:
    env = TrainingTouchBallEnv(spawner_seed=seed)
    max_steps_per_episode = ceil(max_duration_seconds / SIMULATION_STEP_SECONDS)

    os.makedirs(os.path.dirname(log_filename) or ".", exist_ok=True)
    log_file = open(log_filename, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "episode",
        "total_reward",
        "steps",
        "duration_seconds",
        "finish_reason",
        "touched_ball",
    ])

    print(f"Starting training for {num_episodes} episodes. Logging to '{log_filename}'...")
    start_time = time.time()

    try:
        for episode in range(1, num_episodes + 1):
            observation, _ = env.reset(seed=seed)

            blue_reset = getattr(blue_strategy, "reset", None)
            yellow_reset = getattr(yellow_strategy, "reset", None)
            if blue_reset is not None:
                blue_reset(seed=seed)
            if yellow_reset is not None:
                yellow_reset(seed=seed)

            total_reward = 0.0
            episode_steps = 0
            finish_reason = "out_of_time"
            touched_ball = False

            for step in range(1, max_steps_per_episode + 1):
                episode_steps = step

                blue_intents = blue_strategy.decide(observation, env.blue_config)
                yellow_intents = yellow_strategy.decide(observation, env.yellow_config)

                commands = to_rsoccer_commands(observation, env.blue_config, blue_intents)
                commands.extend(
                    to_rsoccer_commands(observation, env.yellow_config, yellow_intents)
                )

                observation, reward, done, _, _ = env.step(commands)
                total_reward += reward

                # Pass per-step reward directly to strategy
                if hasattr(blue_strategy, "record_reward"):
                    blue_strategy.record_reward(reward)

                info = getattr(env, "latest_info", {})
                if info.get("touched_ball", False):
                    touched_ball = True

                if done:
                    finish_reason = info.get("finish_reason", "environment_done")
                    break

            # Trigger optimization pass at episode end
            blue_finish = getattr(blue_strategy, "finish", None)
            if blue_finish is not None:
                blue_finish(total_reward)

            duration_s = round(episode_steps * SIMULATION_STEP_SECONDS, 3)

            if episode % log_freq == 0 or episode == num_episodes:
                csv_writer.writerow([
                    episode,
                    round(total_reward, 4),
                    episode_steps,
                    duration_s,
                    finish_reason,
                    int(touched_ball),
                ])
                log_file.flush()

            if episode % 50 == 0 or episode == num_episodes:
                elapsed = time.time() - start_time
                print(
                    f"Ep {episode}/{num_episodes} | Return: {total_reward:.3f} | "
                    f"Steps: {episode_steps} ({duration_s}s) | Reason: {finish_reason} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    finally:
        log_file.close()
        env.close()
        print(f"Training finished. Logs saved to {log_filename}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless RL training runner for SSL Striker ball-touch task."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of training episodes (default: 10000)",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=DEFAULT_EPISODE_DURATION_SECONDS,
        help="Max duration per episode in seconds (default: 5.0)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--log-filename",
        type=str,
        default="training_stats.csv",
        help="Path to CSV log file for plotting (default: training_stats.csv)",
    )
    parser.add_argument(
        "--blue-strategy",
        default="ssl_strategy_api.strategies.striker:StrikerStrategy",
        help="Blue strategy class as module.path:ClassName",
    )
    parser.add_argument(
        "--yellow-strategy",
        default="ssl_strategy_api.strategies.idle_goalkeeper_rl:IdleGoalkeeperStrategy",
        help="Yellow strategy class as module.path:ClassName",
    )

    args = parser.parse_args()

    blue_strategy = _load_strategy(args.blue_strategy)
    yellow_strategy = _load_strategy(args.yellow_strategy)

    train(
        blue_strategy=blue_strategy,
        yellow_strategy=yellow_strategy,
        num_episodes=args.episodes,
        max_duration_seconds=args.max_duration_seconds,
        log_filename=args.log_filename,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())