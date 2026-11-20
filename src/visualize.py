import json
from pathlib import Path
from typing import Any, Dict, List

from src.agent import QLearningAgent
from src.config import GOAL_POS, GRID_SIZE, START_POS
from src.environment import DynamicGridWorld


def record_greedy_episode(
    env: DynamicGridWorld,
    agent: QLearningAgent,
    max_steps: int = 200,
) -> Dict[str, Any]:
    state = env.reset()
    total_reward = 0.0
    steps = 0
    final_event = "timeout"
    frames: List[Dict[str, Any]] = [
        {
            "step": 0,
            "agent_pos": list(env.agent_pos),
            "agent_theta": env.agent_theta,
            "obstacles": [list(o) for o in env.obstacles],
            "perception": list(state),
            "action": None,
            "reward": 0.0,
            "total_reward": 0.0,
            "event": "start",
        }
    ]

    for _ in range(max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        frames.append(
            {
                "step": steps,
                "agent_pos": list(env.agent_pos),
                "agent_theta": env.agent_theta,
                "obstacles": [list(o) for o in env.obstacles],
                "perception": list(state),
                "action": action,
                "reward": reward,
                "total_reward": total_reward,
                "event": str(info["event"]),
            }
        )
        if terminated:
            final_event = str(info["event"])
            break

    return {
        "episode": "Eval (Greedy)",
        "epsilon": 0.0,
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "final_event": final_event,
        "frames": frames,
    }


def _rolling_mean(values: List[float], window: int = 100, stride: int = 25) -> Dict[str, List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    n = len(values)
    for i in range(0, n, stride):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        xs.append(float(i + 1))
        ys.append(round(sum(chunk) / len(chunk), 3))
    if xs and xs[-1] != n:
        chunk = values[max(0, n - window) : n]
        xs.append(float(n))
        ys.append(round(sum(chunk) / len(chunk), 3))
    return {"x": xs, "y": ys}


def save_html_animation(
    history: Dict[str, Any],
    output_path: str = "training_animation.html",
) -> str:
    rewards_curve = _rolling_mean(history["rewards"], window=100, stride=25)
    success_curve = _rolling_mean(history["successes"], window=100, stride=25)
    collision_curve = _rolling_mean(history["collisions"], window=100, stride=25)

    payload = {
        "grid_size": GRID_SIZE,
        "start_pos": list(START_POS),
        "goal_pos": list(GOAL_POS),
        "snapshots": history.get("snapshots", []),
        "curves": {
            "rewards": rewards_curve,
            "successes": success_curve,
            "collisions": collision_curve,
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<title>Q-Learning Dynamic GridWorld - Training Animation</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --goal: #22c55e;
    --danger: #ef4444;
    --warn: #f59e0b;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }}
  body {{
    margin: 0; padding: 20px;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; align-items: center;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 1.4rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 16px; }}
  .layout {{
    display: grid; grid-template-columns: 460px 520px; gap: 20px;
    max-width: 1000px; width: 100%;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px;
  }}
  canvas {{ background: #0b1120; border-radius: 8px; display: block; margin: 0 auto; }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    margin-top: 12px; justify-content: space-between;
  }}
  button, select {{
    background: #334155; color: var(--text); border: 1px solid #475569;
    padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
  }}
  button:hover {{ background: #475569; }}
  .stats {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 12px;
    font-size: 0.85rem;
  }}
  .stat-box {{
    background: #0f172a; padding: 8px 10px; border-radius: 6px;
    border: 1px solid var(--border);
  }}
  .stat-label {{ color: var(--muted); font-size: 0.75rem; }}
  .stat-val {{ font-weight: 600; font-size: 0.95rem; margin-top: 2px; font-family: monospace; }}
  .legend {{
    display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.78rem;
    color: var(--muted); margin-top: 10px; justify-content: center;
  }}
  .dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
  <h1>Dynamic GridWorld (8x8) — Proceso de Entrenamiento Q-Learning</h1>
  <div class="subtitle">Visualización interactiva de episodios de entrenamiento y curvas de convergencia</div>

  <div class="layout">
    <div class="card">
      <canvas id="gridCanvas" width="420" height="420"></canvas>
      <div class="legend">
        <span><i class="dot" style="background:#38bdf8"></i>Agente (▲▶▼◀)</span>
        <span><i class="dot" style="background:#ef4444"></i>Obstáculo móvil (5)</span>
        <span><i class="dot" style="background:#22c55e"></i>Meta (7,7)</span>
      </div>
      <div class="controls">
        <div>
          <label style="font-size:0.8rem;color:var(--muted)">Episodio:</label>
          <select id="epSelect"></select>
        </div>
        <div>
          <button id="prevBtn">⏮</button>
          <button id="playBtn">⏸ Pause</button>
          <button id="nextBtn">⏭</button>
        </div>
        <div>
          <label style="font-size:0.8rem;color:var(--muted)">Vel:</label>
          <input id="speedRange" type="range" min="1" max="15" value="5" style="width:75px;vertical-align:middle" />
        </div>
      </div>
      <div class="stats">
        <div class="stat-box">
          <div class="stat-label">Paso / Evento</div>
          <div class="stat-val" id="stepVal">0 / start</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Acción / Epsilon (ε)</div>
          <div class="stat-val" id="actionVal">- / 1.0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Recompensa Acumulada</div>
          <div class="stat-val" id="rewVal">0.0</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Estado (x, y, θ)</div>
          <div class="stat-val" id="stateVal">(0,0,E)</div>
        </div>
      </div>
    </div>

    <div class="card" style="display:flex;flex-direction:column;justify-content:space-between">
      <div>
        <div style="font-size:0.9rem;font-weight:600;margin-bottom:8px">Recompensa Media Móvil (ventana = 100 ep.)</div>
        <canvas id="rewardChart" width="480" height="210"></canvas>
      </div>
      <div style="margin-top:14px">
        <div style="font-size:0.9rem;font-weight:600;margin-bottom:8px">Tasa de Éxito vs. Colisión (ventana = 100 ep.)</div>
        <canvas id="rateChart" width="480" height="210"></canvas>
      </div>
      <div class="legend" style="margin-top:8px">
        <span><i class="dot" style="background:#38bdf8"></i>Recompensa media</span>
        <span><i class="dot" style="background:#22c55e"></i>Tasa de Meta (%)</span>
        <span><i class="dot" style="background:#ef4444"></i>Tasa de Colisión (%)</span>
        <span><i class="dot" style="background:#f59e0b"></i>Episodio actual</span>
      </div>
    </div>
  </div>

<script>
const DATA = {json.dumps(payload)};
const ACTION_NAMES = {{0: "ROTATE_LEFT ↺", 1: "ROTATE_RIGHT ↻", 2: "FORWARD ↑"}};
const THETA_NAMES = ["N", "E", "S", "W"];

let epIdx = 0;
let frameIdx = 0;
let playing = true;
let timer = null;

const epSelect = document.getElementById("epSelect");
DATA.snapshots.forEach((snap, idx) => {{
  const opt = document.createElement("option");
  opt.value = idx;
  opt.textContent = `Ep ${{snap.episode}} (R=${{snap.total_reward}}, ${{snap.final_event}})`;
  epSelect.appendChild(opt);
}});

epSelect.addEventListener("change", (e) => {{
  epIdx = parseInt(e.target.value, 10);
  frameIdx = 0;
  renderAll();
}});

document.getElementById("playBtn").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
}});
document.getElementById("prevBtn").addEventListener("click", () => {{
  playing = false;
  document.getElementById("playBtn").textContent = "▶ Play";
  frameIdx = Math.max(0, frameIdx - 1);
  renderAll();
}});
document.getElementById("nextBtn").addEventListener("click", () => {{
  playing = false;
  document.getElementById("playBtn").textContent = "▶ Play";
  const frames = DATA.snapshots[epIdx].frames;
  frameIdx = Math.min(frames.length - 1, frameIdx + 1);
  renderAll();
}});

function drawGrid() {{
  const canvas = document.getElementById("gridCanvas");
  const ctx = canvas.getContext("2d");
  const N = DATA.grid_size;
  const cell = canvas.width / N;
  const snap = DATA.snapshots[epIdx];
  const f = snap.frames[frameIdx];

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < N; r++) {{
    for (let c = 0; c < N; c++) {{
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.strokeRect(c * cell, r * cell, cell, cell);
    }}
  }}

  // Start cell
  ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
  ctx.fillRect(0, 0, cell, cell);
  ctx.fillStyle = "#64748b";
  ctx.font = "bold 11px sans-serif";
  ctx.fillText("START", 6, 16);

  // Goal cell
  const [gr, gc] = DATA.goal_pos;
  ctx.fillStyle = "rgba(34, 197, 94, 0.25)";
  ctx.fillRect(gc * cell, gr * cell, cell, cell);
  ctx.fillStyle = "#22c55e";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("★ GOAL", gc * cell + 4, gr * cell + 30);

  const [ar, ac] = f.agent_pos;
  const theta = f.agent_theta;

  // Obstacles
  f.obstacles.forEach(([or_, oc]) => {{
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.roundRect(oc * cell + 9, or_ * cell + 9, cell - 18, cell - 18, 6);
    ctx.fill();
  }});

  // Agent + orientation triangle
  const cx = ac * cell + cell / 2;
  const cy = ar * cell + cell / 2;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(theta * Math.PI / 2);
  ctx.fillStyle = f.event === "collision" ? "#f59e0b" : "#38bdf8";
  ctx.beginPath();
  ctx.moveTo(0, -cell * 0.34);
  ctx.lineTo(cell * 0.28, cell * 0.28);
  ctx.lineTo(0, cell * 0.14);
  ctx.lineTo(-cell * 0.28, cell * 0.28);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Stats
  document.getElementById("stepVal").textContent = `${{f.step}} / ${{f.event}}`;
  const actName = f.action === null ? "INIT" : ACTION_NAMES[f.action];
  document.getElementById("actionVal").textContent = `${{actName}} | ε=${{snap.epsilon}}`;
  document.getElementById("rewVal").textContent = `${{f.total_reward.toFixed(1)}} (step: ${{f.reward >= 0 ? "+" : ""}}${{f.reward}})`;
  const p = f.perception;
  document.getElementById("stateVal").textContent = `(${{p[0]}},${{p[1]}},${{THETA_NAMES[p[2]]}})`;
}}

function drawLineChart(canvasId, seriesList, yMin, yMax, currentEp) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = {{l: 42, r: 14, t: 14, b: 26}};
  const W = canvas.width - pad.l - pad.r;
  const H = canvas.height - pad.t - pad.b;

  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, W, H);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px monospace";
  [0, 0.5, 1].forEach(t => {{
    const val = yMax - t * (yMax - yMin);
    const y = pad.t + t * H;
    ctx.fillText(val.toFixed(0), 4, y + 4);
    ctx.beginPath();
    ctx.strokeStyle = "rgba(51,65,85,0.5)";
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
  }});

  const maxX = seriesList[0].data.x[seriesList[0].data.x.length - 1] || 1;
  seriesList.forEach(s => {{
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    s.data.x.forEach((x, i) => {{
      const yVal = s.scale ? s.data.y[i] * s.scale : s.data.y[i];
      const px = pad.l + (x / maxX) * W;
      const py = pad.t + ((yMax - yVal) / (yMax - yMin)) * H;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }});
    ctx.stroke();
  }});

  // Current episode vertical line
  const epNum = typeof currentEp === "number" ? currentEp : maxX;
  const ex = pad.l + (epNum / maxX) * W;
  ctx.beginPath();
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.moveTo(ex, pad.t);
  ctx.lineTo(ex, pad.t + H);
  ctx.stroke();
  ctx.setLineDash([]);
}}

function renderAll() {{
  drawGrid();
  const curEp = DATA.snapshots[epIdx].episode;
  drawLineChart("rewardChart", [{{data: DATA.curves.rewards, color: "#38bdf8"}}], -100, 100, curEp);
  drawLineChart("rateChart", [
    {{data: DATA.curves.successes, color: "#22c55e", scale: 100}},
    {{data: DATA.curves.collisions, color: "#ef4444", scale: 100}}
  ], 0, 100, curEp);
}}

function loop() {{
  if (playing) {{
    const frames = DATA.snapshots[epIdx].frames;
    if (frameIdx < frames.length - 1) {{
      frameIdx++;
    }} else {{
      epIdx = (epIdx + 1) % DATA.snapshots.length;
      epSelect.value = epIdx;
      frameIdx = 0;
    }}
    renderAll();
  }}
  const speed = parseInt(document.getElementById("speedRange").value, 10);
  setTimeout(loop, Math.max(40, 450 - speed * 26));
}}

renderAll();
setTimeout(loop, 300);
</script>
</body>
</html>
"""
    out_file = Path(output_path).resolve()
    out_file.write_text(html, encoding="utf-8")
    return str(out_file)

