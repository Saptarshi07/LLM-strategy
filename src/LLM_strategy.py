import numpy as np
import random
import math
from glob import glob
from openai import OpenAI
import json
import requests


api_key = #insert appropriate api-key

prompts = {

    '0': "  No instruction",
    '1': "\"Try to get more points than the other agent.\"",
    '2': "\"Maximize your own number of points.\"",
    '3': "\"Maximize your points, keep in mind the other's welfare.\"",
    '4': "\"Accumulate many points, but do not exploit.\"",
    '5': "\"Exploit the other agent as much as possible.\"",
    '6': "\"Play like a pro.\"",
    '7': "\"Be a saint.\"",
    '8': "\"Think about winning.\"",
    '9': "\"Think about fair outcomes.\""
}


##filepath = 'data/utility-prompts/llama-3-2-3b/M1/3051*'

#original-experiments

def system_prompt(aa,ab,ba,bb,utility_prompt):
    """This function goes into the user message of the LLM explaining the instruction and the goal in last line"""

    return (
            "You are in an interaction with another agent.\n"
            "Your interaction is over several rounds.\n"
            "In each round, you have a choice between two options, L or R.\n"
            "The other agent also has a choice between L or R.\n"
            "The two of you will decide without knowing what the other will choose.\n"
            "Depending on the choices, you receive a reward, which is measured in points.\n"
            f"If both of you choose L, both of you get {aa} points.\n"
            f"If you choose L, and the other agent chooses R, "
            f"you get {ab} points and the other agent gets {ba} points.\n"
            f"If you choose R, and the other agent chooses L, "
            f"you get {ba} points and the other agent gets {ab} points.\n"
            f"If both of you choose R, both of you get {bb} point(s).\n"
            "Your final reward is the total number of points you accumulate in all rounds.\n"
            +(f"Here is your goal: {utility_prompt}\n" if utility_prompt else "\n")
        )




#continuation-probability-experiments. 
#REMEMBER THAT THE VALUE OF EXPROB is w and needs to be changed here, not the file where this function is called.

# def system_prompt(aa,ab,ba,bb,utility_prompt):
#     exprob = 0.5
#     return (
#             "You are in an interaction with another agent.\n"
#             "The interaction is over several rounds.\n"
#             f"After each round the interaction ends with probability {exprob}.\n"
#             "In each round, you have a choice between two options, L or R.\n"
#             "The other agent also has a choice between L or R.\n"
#             "The two of you will decide without knowing what the other will choose.\n"
#             "Depending on the choices, you receive a reward, which is measured in points.\n"
#             f"If both of you choose L, both of you get {aa} points.\n"
#             f"If you choose L, and the other agent chooses R, you get {ab} points and the other agent gets {ba} points.\n"
#             f"If you choose R, and the other agent chooses L, you get {ba} points and the other agent gets {ab} points.\n"
#             f"If both of you choose R, both of you get {bb} point(s).\n"
#             "Your final reward is the total number of points you accumulate in all rounds.\n"
#         )


#one-shot-experiments. Prompt different to w=1.

def system_prompt_one_shot(aa,ab,ba,bb,utility_prompt):
    return (
        "You are in an interaction with another agent.\n"
        "You have a choice between two options, L or R.\n"
        "The other agent also has a choice between L or R.\n"
        "The two of you will decide without knowing what the other will choose.\n"
        "Depending on the choices, you receive a reward, which is measured in points.\n"
        "If both of you choose L, both of you get {aa} points.\n"
        f"If you choose L, and the other agent chooses R, you get {ab} points and the other agent gets {ba} points.\n"
        f"If you choose R, and the other agent chooses L, you get {ba} points and the other agent gets {ab} points.\n"
        f"If both of you choose R, both of you get {bb} point(s).\n"
        "Your reward is the number of points you receive.\n"
        "There is no further interaction with the other agent.\n"
        "Do you choose \"L\” or \"R\"? Give only the character as output.\n"
        "Give no explanation.\n" 
   )

def user_prompt(aa,ab,ba,bb,history):
    """This fuction goes into the user message part of the LLM and explain the specifics of the round"""
    """This function is for the treatment where we estimate memory-1 strategies"""

    payoff_matrix = {("L","L"):aa,("L","R"):ab,("R","L"):ba,("R","R"):bb}
    if history == None:
        """This part is for round 1 behavior"""

        return ("This is the first round.\n" 
                "Do you choose \"L\" or \"R\"? Give only the character as output.\n"
                "Give no explanation.\n"
                )
    else:
        """This part is for estimating continuation behavior, while estimating memory-1 strategies."""

        my_move = history[0]; your_move = history[1]
        return (f"In the previous round, you chose {my_move}, the other agent chose {your_move}.\n"
                f"Therefore, you got {payoff_matrix[(my_move,your_move)]} point(s) and the other agent got {payoff_matrix[(your_move,my_move)]} point(s).\n"
                f"This is a new round.\n"
                "Do you choose \"L\" or \"R\"? Give only the character as output.\n"
                "Give no explanation.\n"
                )
    
def memtwo_user_prompt(aa,ab,ba,bb,history):
    """This fuction goes into the user message part of the LLM and explain the specifics of the round"""
    """This function is for the treatment where we estimate memory-2 strategies"""

    payoff_matrix = {("L","L"):aa,("L","R"):ab,("R","L"):ba,("R","R"):bb}
    if history == None:
        """This part is for round 1 behavior"""

        return ("This is the first round.\n" 
                "Do you choose \"L\" or \"R\"? Give only the character as output.\n"
                "Give no explanation.\n"
                )
    else:    
        if not isinstance(history, list):
            """This part is for round 2 behavior"""

            my_move = history[0]; your_move = history[1]
            return (f"In the first round you chose {my_move}, they chose {your_move}.\n"
                    f"Therefore, you got {payoff_matrix[(my_move,your_move)]} point(s) and they got {payoff_matrix[(your_move,my_move)]} point(s).\n"
                    "This is the second round.\n" 
                    "Do you choose \"L\" or \"R\"? Give only the character as output.\n"
                    "Give no explanation.\n")
        else:
            """This part is for estimating continuation behavior, while estimating memory-2 strategies."""

            my_move_last = history[0][0]; your_move_last = history[0][1]
            my_move_second_last = history[1][0]; your_move_second_last = history[1][1]
            return (f"Two rounds ago, you chose {my_move_second_last}, they chose {your_move_second_last}.\n"
                    f"Therefore, you got {payoff_matrix[(my_move_second_last,your_move_second_last)]} point(s) and they got {payoff_matrix[(your_move_second_last,my_move_second_last)]} point(s).\n"
                    f"In the previous round, you chose {my_move_last}, they chose {your_move_last}.\n"
                    f"Therefore, in the previous round you got {payoff_matrix[(my_move_last,your_move_last)]} point(s) and they got {payoff_matrix[(your_move_last,my_move_last)]} point(s).\n"
                    "This is a new round. Do you choose \"L\" or \"R\"? Give only the character as output.\n" 
                    "Give no explanation.\n"
                    )
        

def LLM_move(model,temperature,goal,history,treatment,aa,ab,ba,bb):
    """This code estimates L or R as the move for an LLM model after a history, 
       given the goal, which is one of the 10 prompts earlier in the code (or could be anything arbitrary)
       Treatment specifies whether we are interested in memory-1 or memory-2.
       The inputs for treatment are "memoryone" and "memorytwo".
       The inputs for model for this code is any gpt model in https://portal.apis.huit.harvard.edu/docs/ais-openai-direct/1/overview
       And any anthropic claude model in https://portal.apis.huit.harvard.edu/docs/ais-bedrock-llm/1/overview
       And any meta model too. (Added Sept 7 by Saptarshi)
       But code below needs to be modified if one needs to use anything other than gpt-4o, gpt-5 or anthropic.claude-sonnet-4-20250514-v1:0
       aa,ab,ba,bb are the game parameters, usually set at 3,0,5,1 the Axelrod values."""
    
    
    if treatment == "memoryone":
        instruction = system_prompt(aa,ab,ba,bb,goal)
        continuationtext = user_prompt(aa,ab,ba,bb,history)
    elif treatment == "memorytwo":
        instruction = system_prompt(aa,ab,ba,bb,goal)
        continuationtext = memtwo_user_prompt(aa,ab,ba,bb,history)
    elif treatment == "one-shot":
        instruction = system_prompt_one_shot(aa,ab,ba,bb,goal)
        continuationtext = ''

    
    if "gpt-4o" in model:
        #api_url = "https://go.apis.huit.harvard.edu/ais-openai-direct/v1" #insert your own URL here if applicable
        api_url = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1"
        client = OpenAI(api_key=api_key,base_url=api_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                    {"role": "system", "content": "Follow specified goals."},
                    {"role": "user", "content":  instruction + continuationtext 
                    }],
            temperature=temperature,seed=42)
        
        play = response.choices[0].message.content.strip()
        return play
    
    elif "gpt-5" in model:

        #api_url = "https://go.apis.huit.harvard.edu/ais-openai-direct/v1" #insert your own URL here if applicable
        api_url = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1"
        client = OpenAI(api_key=api_key,base_url=api_url)
        response = client.responses.create(
            model="gpt-5",
            input= [{ 'role': 'developer', 'content': "Follow specified goals." }, 
                    { 'role': 'user', 'content': instruction + continuationtext }],
            reasoning = {
                "effort": "high"
            },
        )
        
        play = response.output[1].content[0].text
        return play

            
    elif "anthropic" in model:

        api_url = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1/" #insert your own URL here if applicable
        payload = {
            "modelId": model,
            "accept": "application/json",
            "contentType": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1,
                "temperature":temperature,
                "top_p":10**-8,
                "messages": [
                    {"role": "user","content": [{"type": "text","text": instruction + continuationtext}]
                    }
                ]
            }
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        data = response.json()
        reply = data["content"][0]["text"].strip()
        return reply

    elif "llama" in model:

        api_url = "https://go.apis.huit.harvard.edu/ais-bedrock-llm/v1/"
        payload = {
        "modelId": model,
        "accept": "application/json",
        "contentType": "application/json",
        "body": {
            "prompt": instruction + continuationtext,
            "temperature": temperature,
            "max_gen_len": 1,
            "top_p": 10**-8
        }
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key
        }

        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code}, {response.text}")

        data = response.json()

        # Adjust depending on the exact API spec (check if "generation" or "outputs")
        if "generation" in data:
            return data["generation"].strip()
        elif "outputs" in data and len(data["outputs"]) > 0:
            return data["outputs"][0]["text"].strip()
        else:
            raise KeyError(f"Unexpected response format: {data}")
        
    elif "gemini" in model:
      
        api_url = "https://chat.dartmouth.edu/api/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction + continuationtext}
                    ]
                }
            ],
            "generationConfig": {
                "seed": 42,
                "max_tokens": 1,
                "temperature": 0.0
            }
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "bearer " + api_key,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        data = response.json()
        reply = data["choices"][0]["message"]["content"][-1]
        return reply
        
        
    
        
        
