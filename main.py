import retro
import neat


def main():
    env = retro.make(game='SuperMarioBros-Nes')
    env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        print("Action ", action)
        print("Image ", ob.shape, "Reward ", rew, "Done? ", done)
        print("Info ", info)


if __name__ == "__main__":
    main()
