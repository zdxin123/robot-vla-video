# robot-vla-video
基于 Isaac Sim 的人形机器人 VLA 视频驱动动作模仿与意图理解项目。

## 项目目标
- 从真实人类视频提取关键点
- 将视频动作映射到 humanoid 机器人关节动作
- 基于意图编码实现任务级动作生成
- 在 Isaac Sim 中完成高精度仿真验证

## 目录结构
- `video2motion/`：视频到动作模型
- `intent_control/`：意图理解与策略生成
- `scripts/`：数据预处理与运行脚本
- `configs/`：配置文件
- `docker/`：容器环境
- `experiments/`：实验评估脚本
- `isaac_sim_env/`：仿真环境说明

## 当前状态
项目初始化中。
