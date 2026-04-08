import gradio as gr
from env.career_env import CareerEnv
from agent.dqn_agent import DQNAgent

def run_simulation():
    env = CareerEnv()
    agent = DQNAgent(7, 4)

    state = env.reset()
    output = ""

    actions = ["Study", "Dream Path", "Primary Path", "Safe Path"]

    while True:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        cds = env.calculate_cds()
        feasible = env.check_feasibility()

        # ✅ INSIDE LOOP (IMPORTANT)
        output += f"📅 Year {int(state[0])}\n"
        output += f"🎯 Decision: {actions[action]}\n"
        output += f"📈 Skill Level: {round(state[1],2)}\n"
        output += f"📊 Career Score (CDS): {round(cds,2)}\n"
        output += f"⚠️ Feasibility: {'Yes' if feasible else 'No'}\n"
        output += f"💰 Impact Score: {round(reward,2)}\n"
        output += "-----------------------------\n"

        if done:
            output += "\n🚀 FINAL CAREER OUTCOME\n"
            output += f"🎓 Final Skill Level: {round(state[1],2)}\n"
            output += f"📊 Final Decision Score: {round(cds,2)}\n"

            if env.path == 0:
                path_name = "Dream Path (High Risk, High Reward)"
            elif env.path == 1:
                path_name = "Primary Path (Balanced)"
            else:
                path_name = "Safe Path (Low Risk)"

            output += f"🛣️ Recommended Path: {path_name}\n"
            break  # ✅ VERY IMPORTANT

    return output


iface = gr.Interface(
    fn=run_simulation,
    inputs=[],
    outputs="text",
    title="🚀 Advipilot Career Simulation",
    description="AI simulates career decisions using RL + CDS + feasibility."
)

iface.launch()