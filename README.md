## gComm: An environment for investigating generalization in Grounded Language Acquisition
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/117645822-50da8200-b18b-11eb-9bca-4eb7f064e7eb.png" alt="gcomm env"/>
</p>

gComm is a step towards developing a robust platform to foster research in grounded language acquisition in a more challenging and realistic setting. It comprises a 2-d grid environment with a set of agents (a stationary speaker and a mobile listener connected via a communication channel) exposed to a continuous array of tasks in a partially observable setting. The key to solving these tasks lies in agents developing linguistic abilities and utilizing them for efficiently exploring the environment. The speaker and listener have access to information provided in different modalities, i.e. the speaker's input is a natural language instruction that contains the target and task specifications and the listener's input is its grid-view. Each must rely on the other to complete the assigned task, however, the only way they can achieve the same, is to develop and use some form of communication. gComm provides several tools for studying different forms of communication and assessing their generalization.

## Table of contents
* [Getting Started](#getting-started)
* [Baselines](#baselines)
* [Demos](#demos)
* [Additional Features](#additional-features)
* [Publications](#publications)

This repository contains the code for the environment, including the baselines and metrics.

<a name="getting-started"></a>
## Getting Started
To set up environment,

```console
$ git clone https://github.com/SonuDixit/gComm.git
$ python setup.py install  # install setuptools package before running this line
$ cd gComm/
```
Run the following to see if the env works.
```
$ python test_package.py  
```
Input actions manually: <'left', 'right', 'forward', 'backward', 'push', 'pull', 'pickup', 'drop'>

### Important Arguments
Arguments can be found in the file: **gComm/arguments.py**

* Environment arguments
`--grid_size`
`--min_other_objects`
`--max_objects`

* Grammar and Vocabulary arguments
`--type_grammar`
`--transitive_verbs`
`--nouns`
`--color_adjectives`
`--size_adjectives`
`--all_light`
`--keep_fixed_weights`

* RL-framework
`--num_episodes`
`--episode_len`
`--grid_input_type`

* Communication Channel
`comm_type`

* Rendering
`--render_episode`
```console
$ python baselines.py --render_episode
$ python baselines.py --render_episode --wait_time 0.6  # slower rendering (default: 0.3)
```


<a name="baselines"></a>
## Baselines
| Task              | Baseline         | Convergence Rewards  |
|:-----------------:|:----------------:|:--------------------:|
|                   | Simple Speaker   |   0.70               |
|                   | Random Speaker   |   0.40               |
|  **Walk**         | Fixed Speaker    |   0.43               |
|                   | Perfect Speaker  |   0.95               |
|                   | Oracle Listener  |   0.99               |
|                   |                  |                      |
|                   | Simple Speaker   |   0.55               |
|                   | Random Speaker   |   0.19               |
|**Push** & **Pull**| Fixed Speaker    |   0.15               |
|                   | Perfect Speaker  |   0.85               |
|                   | Oracle Listener  |   0.90               |

To run each baseline:

* **Simple Speaker** (Categorical)
```
# walk
$ python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type categorical

# push and pull
$ python baselines.py --type_grammar simple_trans --transitive_verbs push,pull --min_other_objects 2 --max_objects 2 --grid_input_type vector --all_light --num_episodes 400000 --episode_len 10 --comm_type categorical
```

* **Random Speaker**
```
# walk
$ python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 200000 --episode_len 10 --comm_type random

# push and pull
$ python baselines.py --type_grammar simple_trans --transitive_verbs push,pull --min_other_objects 2 --max_objects 2 --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type random
```
* **Fixed Speaker**
```
# walk
$ python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 200000 --episode_len 10 --comm_type fixed

# push and pull
$ python baselines.py --type_grammar simple_trans --transitive_verbs push,pull --min_other_objects 2 --max_objects 2 --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type random
```
* **Perfect Speaker**
```
# walk
$ python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 200000 --episode_len 10 --comm_type perfect

# push and pull
$ python baselines.py --type_grammar simple_trans --transitive_verbs push,pull --min_other_objects 2 --max_objects 2 --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type perfect
```
* **Oracle Listener**
```
# walk
$ python baselines.py --type_grammar simple_intrans --grid_input_type with_target --all_light --num_episodes 200000 --episode_len 10 --comm_type oracle

# push and pull
$ python baselines.py --type_grammar simple_trans --transitive_verbs push,pull --min_other_objects 2 --max_objects 2 --grid_input_type with_target --all_light --num_episodes 300000 --episode_len 10 --comm_type oracle
```

<a name="demos"></a>
## Demos
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/118398805-846d4e80-b65a-11eb-9a40-c281905cb84d.gif" width="300" alt="walk-demo"/>
  <img src="https://user-images.githubusercontent.com/36856187/118398843-a070f000-b65a-11eb-8406-a80c4f8e3ff1.gif" width="300" alt="push-demo"/>
  <img src="https://user-images.githubusercontent.com/36856187/118398896-da41f680-b65a-11eb-9a16-6fe2587d5317.gif" width="300" alt="pull-demo"/>
  <figcaption><pre> 1. WALK ; 2. PUSH; 3. PULL </pre> </figcaption>
</p>


<a name="additional-features"></a>
## Additional Features

### 1. Levels: mazes and obstacles
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/117788916-1f73bc00-b248-11eb-8484-e810a6d88591.png" width="300" alt="obstacles-grid"/>
  ⋅⋅⋅⋅⋅⋅
  <img src="https://user-images.githubusercontent.com/36856187/117788200-74630280-b247-11eb-9018-4b03a6c6ab76.png" width="300" height="375" alt="maze-grid"/>
</p>

* Maze parameters
`--obstacles_flag`
`--num_obstacles`
`--enable_maze`
`--maze_complexity`
`--maze_density`

```
$  python baselines.py --enable_maze --maze_complexity 0.3 --maze_density 0.3 --render_episode 

# test on a bigger grid
$ python test_package.py --enable_maze --maze_density 0.3 --maze_complexity 0.3 --grid_size 8 --max_objects 12 --render_episode
```

### 2. Lights Out
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/125507774-bdc30eb6-8e51-4983-ae5e-e3e6b0abefa1.png" width="500" alt="lights-out"/>
</p>

```
$ python baselines.py --lights_out
$ python baselines.py --lights_out --render_episode  # for rendering
```

### 3. Metrics
* **topsim**: measure compositionality of messages 
```
$ python topsim.py

 ============ protocol: perfectly compositional =============
Concept         Messages  
green box       aa        
blue box        ba        
green circle    ab        
blue circle     bb        
pearson_corr = 1.0 , spearman_corr = 1.0

 ============ protocol: surjective (not injective) =============
Concept         Messages  
green box       ab        
blue box        ba        
green circle    ab        
blue circle     bb        
pearson_corr = 0.6793662204867574 , spearman_corr = 0.694022093788567

 ============ protocol: holistic =============
Concept         Messages  
green box       ba        
blue box        aa        
green circle    ab        
blue circle     bb        
pearson_corr = 0.5 , spearman_corr = 0.5

 ============ protocol: ambiguous language =============
Concept         Messages  
green box       aa        
blue box        aa        
green circle    aa        
blue circle     aa        
pearson_corr = 0.3651483716701107 , spearman_corr = 0.36514837167011077
```

### 4. Other types of communication
* Continuous Messages
```
$ python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type continuous
```
* Binary Messages
```
python baselines.py --type_grammar simple_intrans --grid_input_type vector --all_light --num_episodes 300000 --episode_len 10 --comm_type binary
```



<a name="publication"></a>
## Publications
[1] Rishi Hazra and Sonu Dixit, 2021. ["gComm: An environment for investigating generalization in Grounded Language Acquisition"](https://arxiv.org/pdf/2105.03943.pdf). In NAACL 2021 Workshop: ViGIL.

[2] Rishi Hazra*, Sonu Dixit*, and Sayambhu Sen, 2021. ["Zero-Shot Generalization using Intrinsically Motivated Compositional Emergent Protocols"](https://arxiv.org/pdf/2105.05069.pdf). In NAACL 2021 Workshop: ViGIL.

[3] Rishi Hazra*, Sonu Dixit*, and Sayambhu Sen, 2020. ["*Infinite use of finite means*: Zero-Shot Generalization using Compositional Emergent Protocols"](https://arxiv.org/pdf/2012.05011.pdf).
