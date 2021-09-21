# Straight_drive_using_PPO

## Usage
First I recommend creating a python virtual environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To train from scratch:
```
python main.py
```

To test model:
```
python main.py --mode test --actor_model ppo_actor.pth
```

To train with existing actor/critic models:
```
python main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```

NOTE: to change hyperparameters, environments, etc. do it in [main.py](main.py); I didn't have them as command line arguments because I don't like how long it makes the command.

## How it works

[main.py](main.py) is our executable. It will parse arguments using [arguments.py](arguments.py), then initialize our environment and PPO model. Depending on the mode you specify (train by default), it will train or test our model. To train our model, all we have to do is call ```learn``` function! 

[arguments.py](arguments.py) is what main will call to parse arguments from command line.

[ppo.py](ppo.py) contains our PPO model. 
[network.py](network.py) contains a sample Feed Forward Neural Network we can use to define our actor and critic networks in PPO. 

[eval_policy.py](eval_policy.py) contains the code to evaluating the policy. It's a completely separate module from the other code.


