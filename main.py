import retro
import neat


def main():
    env = retro.make(game='SuperMarioBros-Nes')
    env.reset()
    done = False
    while not done:
        # show gameplay
        env.render()

        # generate a random sample action
        action = env.action_space.sample()
        print("Action ", action)

        # preform the action
        img, rew, done, info = env.step(action)
        print("Image ", img.shape, "Reward ", rew, "Done? ", done)
        print("Info ", info)


if __name__ == "__main__":
    main()
