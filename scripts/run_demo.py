import torch
from video2motion.model import Video2MotionNet
from intent_control.intent_encoder import IntentEncoder
from intent_control.policy import PolicyNet


def main():
    video2motion = Video2MotionNet()
    intent_encoder = IntentEncoder()
    policy = PolicyNet()

    dummy_keypoints = torch.randn(1, 64, 99)
    motion = video2motion(dummy_keypoints)
    intent = intent_encoder(motion)
    action = policy(intent)

    print("Predicted motion:", motion.shape)
    print("Intent embedding:", intent.shape)
    print("Final action:", action.shape)


if __name__ == "__main__":
    main()
