import train
import test

if __name__ == '__main__':
    print("Init Routine")
    train.main()
    # we could put a system wait here
    test.main()
    print("End of Routine")