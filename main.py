from dotenv import load_dotenv

print("Load dot env result:", load_dotenv("./utils/.env"))


def main():
    from model.auxiliary.global_vars import GVars

    GVars.init()

    from run.run import run

    run()


if __name__ == "__main__":
    main()
