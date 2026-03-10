import torch
from video2motion.model import Video2MotionNet


def main():
    model = Video2MotionNet()
    dummy_input = torch.randn(2, 64, 99)
    output = model(dummy_input)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
