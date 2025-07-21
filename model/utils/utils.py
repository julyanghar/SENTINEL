import json
import os


def read_json(file_path: str) -> list[dict] | dict:
    """
    Read JSON file, support formats: .json, .jsonl, .jsonfile
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".json" or ext == ".jsonfile":
        with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".jsonl":
        with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unspported extension {ext} for file: {file_path}")
    return data


def is_dino_sentences_equal(text_1: str, text_2: str) -> bool:
    """
    比较两个用于 Gounding DINO 检测的句子是否相等，不考虑顺序。
    """
    return sorted(text_1.lower().split(".")) == sorted(text_2.lower().split("."))


def ensure_lists(*args: tuple) -> list:
    """
    确保输入的参数都是列表形式，并且第一个参数的长度决定了后续参数的列表长度。

    Args:
        多个参数，每个参数可以是单个元素或列表。
    Returns:
        处理后的参数列表，每个参数都是列表形式。
        第一个参数的长度将决定后续参数的列表长度。
        如果某个参数是单个元素，则会被转换为包含该元素的列表。
        如果某个参数是列表，则保持不变。
    """
    if not args:
        return []

    first_arg: list = args[0] if isinstance(args[0], list) else [args[0]]
    length: int = len(first_arg)

    result = [first_arg] + [
        a
        if isinstance(a, list) and len(a) == length
        else ([a] * length if not isinstance(a, list) else a * (length // len(a)) + a[: length % len(a)])
        for a in args[1:]
    ]

    return result if len(result) > 1 else result[0]


def repeat_n(n, *lists: tuple[list]) -> tuple[list]:
    """
    将输入的每个列表都重复 n 遍，然后返回一个包含这些列表的元组
    """

    def repeat_list(elements: list, n: int) -> list:
        return [elem for elem in elements for _ in range(n)]

    single_arg: bool = len(lists) == 1

    if n <= 1:
        return lists[0] if single_arg else lists

    return repeat_list(lists[0], n) if single_arg else tuple(repeat_list(lst, n) for lst in lists)


def split_n(n: int, *lists: tuple[list]) -> tuple[list[list]]:
    """
    将输入的每个列表都分割为长度为 n 的子列表，然后返回一个包含这些子列表的元组

    示例:
    >>> split_n(2, [1, 2, 3, 4, 5, 6])
    [[1, 2], [3, 4], [5, 6]]

    >>> split_n(3, [1, 2, 3, 4, 5, 6], ['a', 'b', 'c', 'd', 'e', 'f'])
    ([[1, 2, 3], [4, 5, 6]], [['a', 'b', 'c'], ['d', 'e', 'f']])

    注意:
    - 如果 n <= 1，则返回原始列表。
    - 如果传入的列表长度不是 n 的整数倍，则最后一个子列表可能会包含剩余的元素。
    """

    def split(elements: list, n: int) -> list:
        return [elements[i * n : (i + 1) * n] for i in range(len(elements) // n)] if elements else elements

    single_arg: bool = len(lists) == 1

    if n <= 1:
        return lists[0] if single_arg else lists

    return split(lists[0], n) if single_arg else tuple(split(lst, n) for lst in lists)


def maybe_return_ls(force_list: bool, *lists: tuple[list]):
    """
    根据 force_list 参数决定是否返回列表或者其第一个元素
    """

    def maybe_ls(e: list, force_list: bool) -> list:
        return e if force_list or len(e) > 1 else e[0]

    single_arg: bool = len(lists) == 1

    return maybe_ls(lists[0], force_list) if single_arg else tuple(maybe_ls(lst, force_list) for lst in lists)
