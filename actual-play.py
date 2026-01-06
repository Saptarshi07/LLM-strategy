import numpy as np
import random
import math
from glob import glob

from src.LLM_play import * # type: ignore



aa,ab,ba,bb = 3,0,5,1 #PAYOFFS
number_of_rounds = 10 #NUMBER OF ROUNDS
temperature=0.0 #BASE TEMPERATURE FOR MODELS THAT NEED TEMPERATURE.


#please modify accordingly for what works for you. 
modelorder = ["anthropic.claude-sonnet-4-20250514-v1:0",
              #"google.gemma-3-27b-it",
              "google_genai.gemini-2.5-pro",
              #"vertex_ai.gemini-2.5-pro",
              "gpt-4o",
              "gpt-5",
              "meta.llama3-3-70b-instruct-v1:0"
              ]

modelnum = [0,1,2,3,4]

#please change as needed
#data will be changed in this location.
#to see exact prompt we use for these experiments,see src/LLM_play.py
path = 'data-sp/actual-play/stopping-01/'


visited = np.zeros((5,5))
for pl1 in [0,1,2,3,4]:
    for pl2 in [0,1,2,3,4]:
        
        if (visited[pl1][pl2]==0) and (visited[pl2][pl1]==0):
            game_history_p1 = []
            game_history_p2 = []
            model1 = modelorder[pl1]; model2 = modelorder[pl2]
            print(pl1,pl2)
            for round in range(1,number_of_rounds+1):
                p1move = LLM_move(model1,temperature,"",game_history_p1,aa,ab,ba,bb)
                p2move = LLM_move(model2,temperature,"",game_history_p2,aa,ab,ba,bb)
                p2move = p2moves[round-1]
                print(p1move,p2move)
                game_history_p1.append((p1move,p2move)); game_history_p2.append((p2move,p1move)); 
                outcomes = [round, p1move, p2move]
                play = [str(x) for x in outcomes]
                with open(path + f"{pl1}_{pl2}_{aa}{ab}{ba}{bb}.txt", "a") as f:
                    line = " ".join(play) + "\n"
                    f.write(line)
                print('')
        visited[pl1][pl2] += 1




