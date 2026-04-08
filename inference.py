from fastapi import FastAPI
from pydantic import BaseModel
from env.career_env import CareerEnv

app = FastAPI()

env = CareerEnv()

class ActionInput(BaseModel):
    action: int

@app.post("/reset")
def reset():
    global env
    env = CareerEnv()
    state = env.reset()
    return {"state": state.tolist()}

@app.post("/step")
def step(input: ActionInput):
    state, reward, done, _ = env.step(input.action)
    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done
    }

@app.get("/validate")
def validate():
    return {"status": "ok"}
