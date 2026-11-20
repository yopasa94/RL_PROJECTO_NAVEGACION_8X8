import json
from pathlib import Path

from src.agent import QLearningAgent
from src.config import Q_TABLE_ENTRIES, TrainingConfig
from src.environment import DynamicGridWorld
from src.evaluate import evaluate_agent
from src.train import train_agent
from src.visualize import record_greedy_episode, save_html_animation


def main() -> None:
    base_dir = Path(__file__).resolve().parent
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

    assert agent.q_table.size == Q_TABLE_ENTRIES
    history = train_agent(env, agent, config)

    q_table_path = agent.save_q_table(str(base_dir / "q_table.npy"))
    exploited_agent = QLearningAgent(seed=config.seed)
    exploited_agent.load_q_table(q_table_path)

    metrics = evaluate_agent(
        env, exploited_agent, num_episodes=500, max_steps=config.max_steps_per_episode
    )

    greedy_snapshot = record_greedy_episode(
        env, exploited_agent, max_steps=config.max_steps_per_episode
    )
    history["snapshots"].append(greedy_snapshot)
    anim_path = save_html_animation(
        history, output_path=str(base_dir / "training_animation.html")
    )

    report_path = base_dir / "test_results.json"
    report_data = {
        "algorithm": "Tabular Double Q-Learning (blind pose (x, y, theta))",
        "q_table_entries": int(exploited_agent.q_table.size),
        "q_table_shape": list(exploited_agent.q_table.shape),
        "q_table_path": q_table_path,
        "episodes": config.episodes,
        "last_100_train_mean_reward": round(sum(history["rewards"][-100:]) / 100.0, 2),
        "evaluation_500_episodes": metrics,
    }
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print(f"Entradas Q-tabla: {exploited_agent.q_table.size} ({exploited_agent.q_table.shape[0]}x{exploited_agent.q_table.shape[1]})")
    print(f"Q-tabla guardada y cargada desde: {q_table_path}")
    print(f"Recompensa media (últimos 100 ep. entrenamiento): {sum(history['rewards'][-100:]) / 100:.2f}")
    print(f"Evaluación - Recompensa media: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"Evaluación - Tasa de éxito (meta): {metrics['goal_rate'] * 100:.1f}%")
    print(f"Evaluación - Tasa de colisión: {metrics['collision_rate'] * 100:.1f}%")
    print(f"Evaluación - Pasos promedio: {metrics['mean_length']:.1f}")
    print(f"Resultados guardados en: {report_path}")
    print(f"Animación guardada en: {anim_path}")


if __name__ == "__main__":
    main()
