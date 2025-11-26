**How To Set Up Environment for Model Training:**

# 1. Clone our repo and TetrisBattle
- `git clone https://github.com/rganguli0753/RL-Project-4-Tetris-Comparison RLFinalProject`
- `cd RLFinalProject`
- `git clone https://github.com/ylsung/TetrisBattle.git TetrisBattle-master`

# 2. Create & activate venv
- `cd main`
- `python3.10 -m venv venv`
- `source venv/bin/activate` 

# 3. Install dependencies
- `pip install --upgrade pip`
- `pip install -r requirements.txt` 

# 4. Install TetrisBattle as editable package
- `cd ../TetrisBattle-master`
- `pip install -e .`
- `cd ../main`

Now we can train the models.
