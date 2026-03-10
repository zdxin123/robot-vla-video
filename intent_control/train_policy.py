import torch
from intent_control.intent_encoder import IntentEncoder
from intent_control.policy import PolicyNet


def main():
    encoder = IntentEncoder()
    policy = PolicyNet()

    dummy_motion = torch.randn(2, 64, 24)
    z = encoder(dummy_motion)
    action = policy(z)

    print("Intent shape:", z.shape)
    print("Action shape:", action.shape)


if __name__ == "__main__":
    main()
