# Traversability-aware Visual Navigation Foundation Model

## Overview
This project is aimed to enhance navigation foundation model by adding and properly processing traversability segmentation input.

- Course: Modern Methods and Algorithms of Generative Artificial Intelligence, Skoltech, 2024
- Team Members: Sergey Bakulin, Ruslan Babakyan
- Final Presentation: https://docs.google.com/presentation/d/1so2qPeCp6HMNgHJt-02J5tdGujXimT__QEoy5OceyDg/edit?usp=sharing

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Results](#results)
- [Run the Project](#run-the-project)
- [Bibliography](#bibliography)

---

## Problem Statement
A recent trend in robotics research is the development of the various Foundation Models for various robotics applications, like navigation, manipulation, locomotion and others. The key goal of the navigation foundation models is to enable a visual-only navigation method that is suitable for any type of the robot, and such models act as local or intermediate planners. There are two main advantages of this approach: reducing cost and complexity of the robotic systems by substituting classical metric-aware methods, which require expensive sensors like lidars, with the universal model that can work with a single cheap camera. However, the naturally arising question is the reliability and safety of such models when applied in the real world, since they act as a black-box models that rely only on the raw image context. One way to solve this issue is to propose an additional, more interpretable input to such models and a proper way to process it internally.

---

## Results
The SoTA navigation model NoMaD was trained on a self-collected dataset and have shown much better results than the same model, trained on an open access datasets. A new segmentation processing pipeline was created and was used in a new Traversability-aware Visual Navigation Foundation Model structure, that, unlike NoMaD, has additional traversability mask encoder based on Efficient Net-b0. Despite of the fact, that using traversability mask didn't have much effect on the model's accuracy, that opened a wide number of a future direction, in which model can be developed, such as using additional depth input, concatenated with the mask.

---

## Run the Project


### Requirements


### Setup and Installation


### Running the Code


## Bibliography

- NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration - https://arxiv.org/abs/2310.07896
- SAM 2: Segment Anything in Images and Videos - https://arxiv.org/abs/2408.00714
