```
            _ _ _ _                                         __   __          _____ 
           (_) | (_)   /\                                   \ \ / /    /\   |_   _|
  _ __ ___  _| | |_   /  \   _ __ ___  _ __   ___ _ __ ___   \ V /    /  \    | |  
 | '_ ` _ \| | | | | / /\ \ | '_ ` _ \| '_ \ / _ \ '__/ _ \   > <    / /\ \   | |  
 | | | | | | | | | |/ ____ \| | | | | | |_) |  __/ | |  __/  / . \  / ____ \ _| |_ 
 |_| |_| |_|_|_|_|_/_/    \_\_| |_| |_| .__/ \___|_|  \___| /_/ \_\/_/    \_\_____|
                                      | |                                          
                                      |_|                                            
```           

## Building
You only need to compose up the eval environments during training.

```bash
docker compose build
docker compose up -d
docker exec -it {contianer} bash # container = {rl, xai, render, env1, eval_env1, eval_env2}
```

## Train
First start the simulator.
If you want to train without simulated wind, remove it.
Enable 'Direct Actuator Control'.
Then in the rl_agent container run the training script
`python3 train.py`

## Test
First start the simulator.
If you want to test without simulated wind, remove it.
Enable 'Direct Actuator Control'.
Then in the rl_agent container run the testing script
`python3 test.py`

