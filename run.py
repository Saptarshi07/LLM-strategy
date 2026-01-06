import numpy as np
import random
import math
from glob import glob

from src.LLM_strategy import * # type: ignore


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

historyM1 = [("L","L"),("L","R"),("R","L"),("R","R")]
allhistoryM2 = [None] + [k for k in historyM1] + [[k1,k2] for k1 in historyM1 for k2 in historyM1]
allhistoryM1 = [None] + historyM1
firstroundsM2 = [None] + [k for k in historyM1]


#####MAIN

#SET MODEL NAME
#model = "anthropic.claude-sonnet-4-20250514-v1:0"
#model = " "google_genai.gemini-2.5-pro"
model = "gpt-4o"
#model = "gpt-5"
#model = "meta.llama3-3-70b-instruct-v1:0"


#SET PAYOFF VALUES
aa,ab,ba,bb = 3,0,5,1

#IF USING STOPPING PROBABILITY, set W in LLM_strategy.py (see README.md)
#PAYOFF VALUES FOR EQUAL GAINS
#x=0
#aa,ab,ba,bb = 10, 0, 10+x, x

#SET TEMPERATURE
temperature=0.0 #if applies; other parameters hard-coded in LLM_strategy

#SET TREATMENT, we need memory-1 data, memory-2 data or one-shot data.
treatment = "memoryone"
#treatment = "one-shot"
#treatment = "memorytwo"

destination = f'data-sp/{model}/'

for runno in range(0,50):
    print(runno)
    for prompt in ['0']:
        print(prompt)
        outcomes = []
        goal = prompts[prompt];
        
        for history in allhistoryM1:
        #for history in [None]:
        #for history in allhistoryM1:

            play = LLM_move(model,temperature,goal,history,treatment,aa,ab,ba,bb) # type: ignore
            if play=="L":
                outcomes.append(1)
            elif play=="R":
                outcomes.append(0)
            else:
                print("Error"); break;

        outcomes_as_strings = [str(x) for x in outcomes]
        print(outcomes)
        with open(destination + str(aa) + str(ab) + str(ba) + str(bb) + f"-run{runno}.txt", "a") as f:
            line = f"{str(prompt)} " + " ".join(outcomes_as_strings) + "\n"
            f.write(line)
    print('')

