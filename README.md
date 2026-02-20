# Physically-Aware Manipulation Benchmark for Vision-Language-Action Models

**Authors:** Nuriev Kamil, Novikov Egor  

## Overview

This repository provides a benchmark designed to evaluate how well Vision-Language-Action (VLA) models account for physical constraints during robotic manipulation.

The benchmark focuses specifically on assessing:

- Stability awareness
- Smoothness of motion
- Sensitivity to inertia and friction
- Handling of fluid-containing objects
- Safe force application

The goal is to provide standardized evaluation scenarios for comparing baseline and physics-augmented VLA models.

---

## Motivation

Standard manipulation benchmarks primarily measure task completion success.  
However, they often fail to evaluate:

- Physical optimality of motion  
- Safety margins  
- Stability under perturbations  
- Fluid-aware manipulation  

This benchmark addresses these gaps.

---

## Benchmark Design

### Scenario Categories

#### 1. Precision-Critical Manipulation
- Transporting a water-filled glass  
- Moving partially filled containers  
- Delicate placement tasks  

#### 2. Stability-Sensitive Tasks
- Handling objects with high center of mass  
- Avoiding tipping during motion  

#### 3. Force-Constrained Tasks
- Grasping under low-friction conditions  
- Controlled pushing tasks  

---

## Evaluation Metrics

The benchmark includes quantitative metrics such as:

- Task completion rate  
- Spillage amount  
- Maximum tilt angle  
- Acceleration / jerk smoothness  
- Force peaks  
- Grasp stability  

Each metric is designed to capture physical sensitivity rather than just binary success.

---

## Usage

1. Deploy benchmark scenarios in simulation or real robot setup.
2. Run candidate VLA model.
3. Collect metric logs.
4. Generate standardized evaluation report.

---

## Output

The benchmark produces:

- Structured metric logs
- Comparative performance tables
- Stability and smoothness statistics
- Aggregated physical optimality score

---

## Target Systems

- UR10e robotic arm (recommended)
- RGB camera
- Parallel gripper

The benchmark can also be adapted to other robotic platforms.

---

## Contribution

This benchmark enables reproducible and structured evaluation of physically-aware robotic manipulation, facilitating fair comparison between baseline and physics-augmented VLA models.
