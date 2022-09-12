# Hierarchical Approach for Joint Semantic, Plant Instance, and Leaf Instance Segmentation in the Agricultural Domain

This repo contains the code of the paper *Hierarchical Approach for Joint Semantic, Plant Instance, and Leaf Instance Segmentation in the Agricultural Domain*, by [G. Roggiolani](https://github.com/theroggio) and [M. Sodano](https://github.com/matteosodano) et al., submitted to the IEEE International Conference on Robotics and Automation (ICRA) 2023.

## Abstract
Plant phenotyping is a central task in agriculture, as it describes plants’  growth stage, development, and other relevant quantities. Robots can help automate this process by accurately estimating plant traits such as the number of leaves, leaf area, and the plant size. In this paper, we address the problem of joint semantic, plant instance, and leaf instance segmentation of crop fields from RGB data. We propose a single convolutional neural network that addresses the three tasks simultaneously, exploiting their underlying hierarchical structure. We introduce task-specific skip connections, which our experimental evaluation proves to be more beneficial than the usual schemes. We also propose a novel automatic post-processing, which explicitly addresses the problem of spatially close instances, common in the agricultural domain because of overlapping leaves. Our architecture simultaneously tackles these problems jointly in the agricultural context. Previous works either focus on plant or leaf segmentation, or do not optimise for semantic segmentation. Results show that our system has superior performance to state-of-the-art approaches, while having a reduced number of parameters and is operating at camera frame rate.

## Results
Coming soon!

## Code
Coming soon!

## Citation
If you use our framework for any academic work, please cite the original [paper]().

> bibtex entry coming soon!

## Acknowledgment
This work has partially been funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy, EXC-2070 -- 390732324 -- PhenoRob, and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under STA~1051/5-1 within the FOR 5351~(AID4Crops).