# LLM-strategy
Accompanying python script for manuscript by Pal et al. 

The src folder contains three files: LLM_play, LLM_strategy and LLM_tools.

LLM_strategy.py:

-LLM_strategy.py covers the code for original experiment, stopping probability, framings, equal gains from switching and memory-2.
-To make sure we are running original/framings/equal gains/memory-2, please use the first system_prompt function in LLM_strategy (lines 31-49), keeping the other one (line 57-72) commented out. 
-For the stopping probability, use the latter function and comment out the first one. 
-The stopping probability value is set within this function under `exprob'.
-The actions L or R for the LLM are obtained by running the code in run.py, in the outer folder. 
-There we can set the model name, payoff values and temperature of the LLM, if applies. Other parameters of the LLMs are hard-coded in LLM_strategy functions.
- Finally in run.py, you can set the treament: memory-1, memory-2 or one-shot; depending on which one you choose, you select a different for-loop in line 63-65.

LLM_play.py:

-This contains all the functions necessary for simulating a real game between two LLMs.
-In line 34-36 you can choose what prompt to precisely use "lasts 10 rounds", "lasts atleast 10 rounds" and the stopping probability treatment
-The game is executed by running actual-play.py, where all games run for 10 rounds (variable number_of_rounds).
-

LLM_tools.py:

-Contains all the function necessay for a) computing payoffs, b) checking if memory-1 or 2 strategy is Nash, Partner or Rival, c) Percentage of random memory-1 or two strategies that earn more than the self-payoff.
-d) Drawing the convex hulls of payoffs 

For any question, contact spal@math.harvard.edu

The generated data from this code can be found in Zenodo, under the DOI 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18148157.svg)](https://doi.org/10.5281/zenodo.18148157)

