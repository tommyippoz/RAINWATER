# RAINWATER
fRamework for crAftINg poWer consumption Anomaly deTEcors from ScRatch

## Purpose

Monitoring power grid infrastructures typically generates a massive amount of power consumption data related to different components or communication channels to be used for planning power optimization. 
However, it does not suffice for conducting other key tasks for attaining desirable properties such as reliability, safety and security. 
In these cases, the grid should be monitored for detecting anomalies due to security threats, component failures, environmental damages, or other hazards. 
Even relying on the most sophisticated unsupervised Machine Learning techniques, learning how to detect anomalies using only normal data is not enough to deriving a solution that is good enough to be deployed in power infrastructures. 
To tackle this problem, this paper analyzes the state of the art of existing solutions for power consumption anomaly detection and threat and anomaly models for smart grids. 
Building on that, the paper proposes the following novel contributions: i) a generic and comprehensive model that links threats to power grid infrastructures to potential effects (anomalies),ii) derives a total of 6 power consumption anomalies that can be injected into normal data traces through false data injection, iii) proposes a software architecture and a supporting library for automatically crafting anomaly detectors, which is tested in iv) and v); iv) a case study from the industrial domain, and v) multiple publicly available datasets related to grid infrastructures. 
Expertise from academia and industrial partners allows for providing findings that have a major relevance for the research landscape and can be directly and immediately useful for stakeholders and companies.

## Dependencies

You will need standard libraries for machine learning and tabular data
- NumPy
- Pandas
- Scikit-Learn
- xgboost
and related cascading dependencies

## Usage

An usage example can be seen [here](simple_main.py).
Note that connecting to any dataset requires to specify a configuration file that tells the library how to fetch data.
These configuration files have to be structures as shown in the [example](input/RS2DG.cfg), and placed in the 'input' folder.
Should you need a different setup for paths and for general data about analyses, just edit the [general configuration file](general_cfg.cfg).