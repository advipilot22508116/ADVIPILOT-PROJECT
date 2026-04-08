from fastapi import FastAPI
from env.career_env import CareerEnv

app = FastAPI()

env = CareerEnv()

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.tolist()}

@app.post("/step")
def step(action: int):
    state, reward, done, _ = env.step(action)
    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done
    }

@app.get("/validate")
def validate():
    return {"status": "ok"}