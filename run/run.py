if __name__ == "__main__":
    print("Please run main.py")
    exit(0)


def run() -> None:
    from accelerate import Accelerator, find_executable_batch_size

    from model.auxiliary.dataset import DataSet
    from model.auxiliary.global_vars import GVars
    from run.build_dataset import run_build_dataset

    args, save_path, logger = GVars.args, GVars.save_path, GVars.logger
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def dynamic_batch_size_func(batch_size):
        nonlocal accelerator
        accelerator.clear()

        logger.info(f"Current batch size: {batch_size}")
        logger.info(f"Start loading dataset with dataset path: {args.dataset_path}")
        dataset: DataSet = DataSet(args=args, logger=logger)
        dataset.filter(save_path)
        logger.info(f"Finish loading dataset, dataset size: {len(dataset.data)}")

        if not dataset.data:
            logger.info("All data has been processed, exit run function")
            return
        run_build_dataset(dataset.data, batch_size)

    dynamic_batch_size_func()
