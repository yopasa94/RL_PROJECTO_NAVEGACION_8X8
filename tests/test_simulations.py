import json
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import QLearningAgent
from src.config import EAST, SOUTH, TrainingConfig
from src.environment import DynamicGridWorld
from src.evaluate import evaluate_agent
from src.train import train_agent
from src.visualize import record_greedy_episode, save_html_animation


def run_multiple_simulations(
    env: DynamicGridWorld,
    agent: QLearningAgent,
    num_simulations: int = 20,
    max_steps: int = 200,
) -> List[Dict[str, Any]]:
    simulations: List[Dict[str, Any]] = []
    theta_labels = {EAST: "EAST→SOUTH", SOUTH: "SOUTH→EAST"}
    for sim_id in range(1, num_simulations + 1):
        snap = record_greedy_episode(env, agent, max_steps=max_steps)
        init_theta = snap["frames"][0]["agent_theta"]
        dir_label = theta_labels.get(init_theta, str(init_theta))
        snap["episode"] = f"Sim #{sim_id} ({dir_label})"
        simulations.append(snap)
    return simulations


def test_final_policy_in_many_simulations() -> None:
    config = TrainingConfig()
    env = DynamicGridWorld(seed=config.seed)
    agent = QLearningAgent(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed,
    )

    history = train_agent(env, agent, config)

    q_table_file = Path(__file__).resolve().parent.parent / "q_table.npy"
    saved_path = agent.save_q_table(str(q_table_file))
    loaded_agent = QLearningAgent(seed=config.seed)
    loaded_agent.load_q_table(saved_path)

    metrics_1000 = evaluate_agent(
        env,
        loaded_agent,
        num_episodes=1000,
        max_steps=config.max_steps_per_episode,
    )
    if env.simultaneous:
        assert metrics_1000["goal_rate"] >= 0.50
        assert metrics_1000["mean_reward"] >= -10.0
    else:
        assert metrics_1000["goal_rate"] >= 0.90
        assert metrics_1000["mean_reward"] > 0.0

    simulations = run_multiple_simulations(
        env,
        loaded_agent,
        num_simulations=20,
        max_steps=config.max_steps_per_episode,
    )

    sim_history = {
        "rewards": history["rewards"],
        "successes": history["successes"],
        "collisions": history["collisions"],
        "snapshots": simulations,
    }
    out_html = Path(__file__).resolve().parent / "simulations_animation.html"
    anim_path = save_html_animation(sim_history, output_path=str(out_html))

    report_file = Path(__file__).resolve().parent / "simulations_report.json"
    report_payload = {
        "q_table_path": saved_path,
        "metrics_1000_episodes": metrics_1000,
        "recorded_simulations": [
            {
                "episode": s["episode"],
                "final_event": s["final_event"],
                "steps": s["steps"],
                "total_reward": s["total_reward"],
            }
            for s in simulations
        ],
    }
    report_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("=== Resultados en 1000 Simulaciones (Orientación Inicial EAST o SOUTH) ===")
    print(f"Recompensa media: {metrics_1000['mean_reward']:.2f} +/- {metrics_1000['std_reward']:.2f}")
    print(f"Tasa de éxito (meta): {metrics_1000['goal_rate'] * 100:.1f}%")
    print(f"Tasa de colisión: {metrics_1000['collision_rate'] * 100:.1f}%")
    print(f"Pasos promedio: {metrics_1000['mean_length']:.1f}")
    print("\n=== Detalle de las primeras 10 Simulaciones Grabadas ===")
    for sim in simulations[:10]:
        print(
            f"  {sim['episode']:<22} | Evento: {sim['final_event']:<6} "
            f"| Pasos: {sim['steps']:<3} | Recompensa: {sim['total_reward']:.1f}"
        )
    print(f"\nAnimación de las 20 simulaciones guardada en: {anim_path}")


if __name__ == "__main__":
    test_final_policy_in_many_simulations()
