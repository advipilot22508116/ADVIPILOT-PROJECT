import gradio as gr
import matplotlib.pyplot as plt
from env.career_env import CareerEnv
from agent.dqn_agent import DQNAgent

def run_simulation(years):
    env = CareerEnv()
    agent = DQNAgent(7, 4)

    state = env.reset()

    actions = ["Study", "Dream Path", "Primary Path", "Safe Path"]

    output = "## 📊 Career Simulation Log\n\n"

    skills = []
    cds_list = []

    for step in range(years):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        cds = env.calculate_cds()
        feasible = env.check_feasibility()

        skills.append(round(state[1], 2))
        cds_list.append(round(cds, 2))

        # Only show every 2nd year (clean output)
        if step % 2 == 0:
            output += f"**Year {int(state[0])}**\n"
            output += f"- 🎯 Decision: {actions[action]}\n"
            output += f"- 📈 Skill: {round(state[1],2)}\n"
            output += f"- 📊 CDS: {round(cds,2)}\n"
            output += f"- ⚠️ Feasible: {'Yes' if feasible else 'No'}\n"
            output += f"- 💰 Reward: {round(reward,2)}\n\n"

        if done:
            break

    # Final summary
    output += "## 🚀 Final Outcome\n"
    output += f"- 🎓 Final Skill: {round(state[1],2)}\n"
    output += f"- 📊 Final CDS: {round(cds,2)}\n"

    if cds > 0.7:
        output += "- ✅ Strong career trajectory\n"
    else:
        output += "- ⚠️ Needs improvement\n"

    # Plot graph
    plt.figure()
    plt.plot(skills)
    plt.title("Skill Growth Over Time")
    plt.xlabel("Years")
    plt.ylabel("Skill Level")

    return output, plt


iface = gr.Interface(
    fn=run_simulation,
    inputs=gr.Slider(5, 20, value=10, label="Simulation Years"),
    outputs=["markdown", "plot"],
    title="🚀 Advipilot Career Simulation",
    description="AI simulates career decisions using Reinforcement Learning + CDS + Feasibility"
)

iface.launch(share=True)
