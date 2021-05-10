# gComm: 
## An environment for investigating generalization in Grounded Language Acquisition
![env_desc](https://user-images.githubusercontent.com/36856187/117645822-50da8200-b18b-11eb-9bca-4eb7f064e7eb.png)

gComm is a step towards developing a robust platform to foster research in grounded language acquisition in a more challenging and realistic setting. It comprises a 2-d grid environment with a set of agents (a stationary speaker and a mobile listener connected via a communication channel) exposed to a continuous array of tasks in a partially observable setting. The key to solving these tasks lies in agents developing linguistic abilities and utilizing them for efficiently exploring the environment. The speaker and listener have access to information provided in different modalities, i.e. the speaker's input is a natural language instruction that contains the target and task specifications and the listener's input is its grid-view. Each must rely on the other to complete the assigned task, however, the only way they can achieve the same, is to develop and use some form of communication. gComm provides several tools for studying different forms of communication and assessing their generalization.

This repository contains the code for the environment, including the baselines and metrics.

## Setup
To run this project,

```
$ cd ../lorem
$ npm install
$ npm start
```
