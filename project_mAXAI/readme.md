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

## Specify environments
Specify number of training or testing environements and number of evaluation environments. Evaluation environments are used for training purposes.
Execute the python script
`.\customize_num_envs.py x y`
where x is the number of environments and y is the number of evaluation environments

## Building
Make sure you have the correct number of env(s) and eval_env(s).
You only need to compose up the eval environments during training.

```bash
docker compose build
docker compose up -d
docker exec -it {contianer} bash # container = {drl, xai, render, env1, eval_env1, eval_env2}
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

WORKDIR /workspace
RUN /bin/bash -c "source devel/setup.bash" 
# && roslaunch src/simulator.launch && rosservice call /supervisor/switch_mode "mode: 'direct_actuator_control'" "

rosservice call /sim_vessel/reset_state 334.61 990.24 0 0 0 0 0 0 0 0 0 0 0 0 && rostopic pub /actuator_ref_1 custom_msgs/ActuatorSetpoints "{throttle_reference: 0, angle_reference: 135}" && rostopic pub /actuator_ref_2 custom_msgs/ActuatorSetpoints "{throttle_reference: 0, angle_reference: -135}" && rostopic pub /actuator_ref_3 custom_msgs/ActuatorSetpoints "{throttle_reference: 0, angle_reference: -45}" && rostopic pub /actuator_ref_4 custom_msgs/ActuatorSetpoints "{throttle_reference: 0, angle_reference: 45}" && rosservice call /sim_vessel/reset_state 334.61 990.24 0 0 0 0 0 0 0 0 0 0 0 0