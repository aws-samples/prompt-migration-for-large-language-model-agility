from deepeval.metrics.utils import (
    trimAndLoadJson,
)
import time

def call_trim_and_load(model, prompt):
    res = model.generate(prompt)
    try: 
        data = trimAndLoadJson(res)
        return data
    except: 
        print("ERROR in output, trying again")
        print(res)
        return call_trim_and_load(model, prompt)
    
async def a_call_trim_and_load(model, prompt):
    res = await model.a_generate(prompt)
    try: 
        data = trimAndLoadJson(res)
        return data
    except: 
        print("ERROR in output, trying again")
        return await a_call_trim_and_load(model, prompt)