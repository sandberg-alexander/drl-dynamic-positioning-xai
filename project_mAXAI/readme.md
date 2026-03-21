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
      
## Specify environment
Specify what environment the DRL agents should use. Step 1 is to run `python3 genertate_compose.py` with the arguments `local` or `remote` to generate a docker-compose.yml. 

### Local environement
```bash
python3 genertate_compose.py local
```
sets up a ROS connection using a bridge network on the host machine to access a local simulator. Mainly used for training, evaluating, testing and developing DRL and XAI scripts and models.

### Remote environment
```bash
python3 genertate_compose.py remote
```
sets up a ROS connection over LAN that can either be a remote simulator, or the real *milliAmpere1* interface. Mainly used for Hardware In Loop (HIL) experiments and deployment. Executing the command in remote mode must also include the flag `--remote-ip {remote_ip}` which specifies the IP address of the remote host running the *milliAmpere1* code. The local IP will attempt to auto connect, if it fails this can be specified with `--local-ip {local_ip}`.

The remote host must set some environmental variables. This can be done manually or by building and running the containers using the `project/2025/xai` branch of the *milliAmpere1* code. Generally it might not be possible to build the containers from scratch. To set them manually, execute the following where `{remote_ip}` is the local IP of the remote host system.
```bash
export ROS_MASTER_URI = http://{remote_ip}:11311
export ROS_IP = {remote_ip}
source ~/.bashrc
```

## Be on correct branch
Make sure that the *milliAmpere1* code is on the `project/2025/xai` branch. To change the branch, do
```bash
git checkout project/2025/xai
```
if the git submodule is not loaded inn, do
```bash
git submodule init
git submodule update
```
and then switch to the correct branch.

## Building, starting & stopping the containers
Build the container images and start the containers by doing
```bash
docker compose build
docker compose up -d
docker exec -it {container} bash # container = {drl, xai, simulator_local}
```
if using a remote environment, make sure the container is running on the remote host machine.

To stop the containers run
```bash
docker compose down --remove-orphans -t 0 
```

## Enable X11 Forwarding
Before using the DRL agent, or XAI dashboard, make sure that X11 Forwarding is enabled. This must be enabled every time the user logs in to the computer by doing
```bash
xhost +
```

## Train DRL agent
Make sure you are in the local environment and start the containers.
Go to `0.0.0.0:61200` in a browser and select the DRL mode from the drop down menu, or do
```bash
rosservice call /supervisor/switch_mode "mode: 'drl'"
```
in the simulator container. Then in the drl container run the training script
```bash
python3 train.py
```

## Evaluating DRL agent
Make sure the simulator is running, similar to the training scenario.
Start evaluating by executing
```bash
./run_evaluation.sh
```
This evaluator will go through all models saved by the training module, starting from the back. It also remembers the progress, so the evaluator can be stopped and re-started at any time. Go out of project_mAXAI and into local_env and run `python3 plot_eval.py` to find the best model.

## Deploying DRL agent
To deploy an agent, do  
```bash
python3 deploy.py
```
This will open a black screen that takes in inputs, which is described in the terminal.

## Starting XAI Dashboard
To start the Dashboard, make sure the simulator is running, is in DRL mode and the agent is deployed. Then go to the xai container and do
```bash
python3 shap_explanation.py
```
This will open the Dashboard. The same inputs in the deploy window can be done in the dashboard.

## Turn on simulated wind
To turn on the simulated wind in the simulator, make sure the simulator is running and open the simulator container and execute
```bash
rosservice call /sim/set_wind_sigma "steady: 3.0
gust: 5.0"
```
You can change the values to whatever you want.

