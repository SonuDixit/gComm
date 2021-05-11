## gComm: An environment for investigating generalization in Grounded Language Acquisition
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/117645822-50da8200-b18b-11eb-9bca-4eb7f064e7eb.png" alt="gcomm env"/>
</p>

gComm is a step towards developing a robust platform to foster research in grounded language acquisition in a more challenging and realistic setting. It comprises a 2-d grid environment with a set of agents (a stationary speaker and a mobile listener connected via a communication channel) exposed to a continuous array of tasks in a partially observable setting. The key to solving these tasks lies in agents developing linguistic abilities and utilizing them for efficiently exploring the environment. The speaker and listener have access to information provided in different modalities, i.e. the speaker's input is a natural language instruction that contains the target and task specifications and the listener's input is its grid-view. Each must rely on the other to complete the assigned task, however, the only way they can achieve the same, is to develop and use some form of communication. gComm provides several tools for studying different forms of communication and assessing their generalization.

## Table of contents
* [General info](#general-info)
* [Getting Started](#getting-started)
* [Baselines](#baselines)
* [Levels](#levels)
* [Metrics](#metrics)
* [Additional Features](#additional-features)
* [Publications](#publications)

This repository contains the code for the environment, including the baselines and metrics.

## Getting Started
To run this project,

```
$ git clone https://github.com/SonuDixit/gComm.git
$ python setup.py install
```

### Important Arguments
Arguments can be found in the file: gComm/arguments.py

* Environment arguments
`grid_size`
`min_other_objects`
`max_objects`

* Grammar and Vocabulary arguments
`type_grammar`
`transitive_verbs`
`nouns`
`color_adjectives`
`size_adjectives`
`all_light`
`keep_fixed_weights`


## Baselines
| Task              | Baseline         | Convergence Rewards  |
|:-----------------:|:----------------:|:--------------------:|
|                   | Simple Speaker   |   0.80               |
|                   | Random Speaker   |   0.40               |
|  **Walk**         | Fixed Speaker    |   0.43               |
|                   | Perfect Speaker  |   0.95               |
|                   | Oracle Listener  |   0.99               |
|                   |                  |                      |
|                   | Simple Speaker   |   0.70               |
|                   | Random Speaker   |   0.19               |
|**Push** & **Pull**| Fixed Speaker    |   0.15               |
|                   | Perfect Speaker  |   0.85               |
|                   | Oracle Listener  |   0.90               |

To run each baseline:

* **Random Speaker**
```
$ python  # walk
$ python  # push and pull
```
* **Fixed Speaker**
```
$ python  # walk
$ python  # push and pull
```
* **Perfect Speaker**
```
$ python  # walk
$ python  # push and pull
```
* **Oracle Listener**
```
$ python  # walk
$ python  # push and pull
```

## Levels
<p align="center">
  <img src="https://user-images.githubusercontent.com/36856187/117788916-1f73bc00-b248-11eb-8484-e810a6d88591.png" width="300" alt="obstacles-grid"/>
  ⋅⋅⋅⋅⋅⋅
  <img src="https://user-images.githubusercontent.com/36856187/117788200-74630280-b247-11eb-9018-4b03a6c6ab76.png" width="300" height="375" alt="maze-grid"/>
</p>

* Maze parameters
`obstacles_flag`
`num_obstacles`
`enable_maze`
`maze_complexity`
`maze_density`

## Lights Out

Set `lights_out` argument to True

## Publications
[1] Rishi Hazra and Sonu Dixit, 2021. ["gComm: An environment for investigating generalization in Grounded Language Acquisition"](https://arxiv.org/pdf/2105.03943.pdf). In NAACL 2021 Workshop: ViGIL.

[2] Rishi Hazra*, Sonu Dixit*, and Sayambhu Sen, 2021. ["Zero-Shot Generalization using Intrinsically Motivated Compositional Emergent Protocols"](). In NAACL 2021 Workshop: ViGIL.

[3] Rishi Hazra, Sonu Dixit, and Sayambhu Sen, 2020. ["*Infinite use of finite means*: Zero-Shot Generalization using Compositional Emergent Protocols"](https://arxiv.org/pdf/2012.05011.pdf).
