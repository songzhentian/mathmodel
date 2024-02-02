# tennis
A MATH4581 Project. Determines the odds that two players may win a tennis match based on skill using Markov chains.

# how to run
Install Python and numpy onto your machine.
Download `tennis.py` to your computer
Open a terminal and navigate to where you saved `tennis.py`
Type the command `python tennis.py <prob_a> <prob_b>` where prob is the odds that a player wins a turn they serve, respectively
The output will be [probability A wins, probability B wins]

# example outputs

`python tennis.py 0.5 0.5` -> [0.5, 0.5]

`python tennis.py 0.49 0.5` -> [0.47586568920443906, 0.5241343107955609]

`python tennis.py 0.4 0.5` -> [0.28008402173184277, 0.7199159782681572]

`python tennis.py 0.6 0.5` -> [0.7312376455945588, 0.26876235440544116]

`python tennis.py 0.7 0.5` -> [0.8869863499622428, 0.11301365003775721]
