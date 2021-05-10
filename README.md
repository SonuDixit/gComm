## gComm: An environment for investigating generalization in Grounded Language Acquisition
![env_desc](https://user-images.githubusercontent.com/36856187/117645822-50da8200-b18b-11eb-9bca-4eb7f064e7eb.png)

gComm is a step towards developing a robust platform to foster research in grounded language acquisition in a more challenging and realistic setting. It comprises a 2-d grid environment with a set of agents (a stationary speaker and a mobile listener connected via a communication channel) exposed to a continuous array of tasks in a partially observable setting. The key to solving these tasks lies in agents developing linguistic abilities and utilizing them for efficiently exploring the environment. The speaker and listener have access to information provided in different modalities, i.e. the speaker's input is a natural language instruction that contains the target and task specifications and the listener's input is its grid-view. Each must rely on the other to complete the assigned task, however, the only way they can achieve the same, is to develop and use some form of communication. gComm provides several tools for studying different forms of communication and assessing their generalization.

## Table of contents
* [General info](#general-info)
* [Getting Started](#getting-started)
* [Baselines](#baselines)
* [Levels](#levels)
* [Metrics](#metrics)
* [Publications](#publications)

This repository contains the code for the environment, including the baselines and metrics.

## Getting Started
To run this project,

```
$ cd ../lorem
$ npm install
$ npm start
```

## Baselines
| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

## Publications
[1] Rishi Hazra and Sonu Dixit, 2021. ["gComm: An environment for investigating generalization in Grounded Language Acquisition"](). In NAACL 2021 Workshop: ViGIL.

[2] Rishi Hazra, Sonu Dixit, and Sayambhu Sen, 2021. ["Zero-Shot Generalization using Intrinsically Motivated Compositional Emergent Protocols"](). In NAACL 2021 Workshop: ViGIL

[3] Rishi Hazra, Sonu Dixit, and Sayambhu Sen, 2020. ["*Infinite use of finite means*: Zero-Shot Generalization using Compositional Emergent Protocols"](https://arxiv.org/pdf/2012.05011.pdf).
