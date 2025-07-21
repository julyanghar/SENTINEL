import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import nltk
import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

BASE_URL = "YOUR BASE URL HERE"
API_KEY = "YOUR API KEY HERE"

NUM_SECONDS_TO_SLEEP = 0.5
MAX_RETRIES = 3


class OpenAIModel:
    def __init__(
        self,
        base_url=BASE_URL,
        api_key=API_KEY,
        model: str | None = None,
        timeout_sec: int = 20,
    ):
        if not base_url.endswith("/v1/"):
            if base_url.endswith("/"):
                base_url = base_url[:-1]
            base_url = f"{base_url}/v1/"
        self.client: OpenAI = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
            max_retries=MAX_RETRIES,
        )
        self.model: str | None = model

    def _u_a_to_messages(self, users: list[str], systems: list[str]) -> list[list[dict]]:
        assert len(users) == len(systems), "Length of users and systems must be the same."
        messages_list: list[list[dict]] = []
        for u, s in zip(users, systems):
            messages: list[dict] = []
            if s:
                messages.append({"role": "system", "content": s})
            if u:
                messages.append({"role": "user", "content": u})
            messages_list.append(messages)
        return messages_list

    def _prepare_messages_list(
        self,
        users: list[str] | str | None,
        systems: list[str] | str | None,
        messages: list[list[dict]] | list[dict] | None,
    ) -> list[list[dict]]:
        def ensure_lists(*args) -> list:
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

        if users is not None:
            users, systems = ensure_lists(users, systems)
            return self._u_a_to_messages(users, systems)
        elif systems is not None:
            systems, users = ensure_lists(systems, users)
            return self._u_a_to_messages(users, systems)
        else:
            if not isinstance(messages[0], list):
                return [messages]
        return messages

    def gen(
        self,
        users: list[str] | str | None = None,
        systems: list[str] | list | None = None,
        messages: list[list[dict]] | list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        model: str | None = None,
        sample=False,
        force_list: bool = False,
        return_completions: bool = False,
        use_parallel: bool = True,
        use_tqdm: bool = False,
        max_workers: int = 64,
    ) -> list[str] | str:
        assert (users is None and systems is None) == (messages is not None), "Invalid input arguments."

        messages: list[list[dict]] = self._prepare_messages_list(users, systems, messages)

        n = 1 if not sample else 5
        outputs: list[str | list[str]] = [None] * len(messages)

        def gen_completion(messages: list[dict]) -> str | list[str]:
            completions: ChatCompletion = self._gen(
                messages=messages,
                model=model if model else self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )

            if return_completions:
                return completions
            if len(completions.choices) == 1:
                return completions.choices[0].message.content.strip()
            else:
                return [choice.message.content.strip() for choice in completions.choices]

        if len(messages) == 1:
            use_tqdm = False

        if use_tqdm:
            pb = tqdm(total=len(messages))

        if use_parallel and len(messages) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future, int] = {executor.submit(gen_completion, m): i for i, m in enumerate(messages)}

                for completed_future in as_completed(futures):
                    if use_tqdm:
                        pb.update(1)
                    index: int = futures[completed_future]
                    outputs[index] = completed_future.result()
        else:
            for i, m in enumerate(messages):
                outputs[i] = gen_completion(m)
                if use_tqdm:
                    pb.update(1)

        return outputs if force_list or len(outputs) > 1 else outputs[0]

    def _gen(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        model: str | None = None,
        temperature: float = 0.2,
        n: int = 1,
        seed: int | None = None,
        top_p: float = 1.0,
    ) -> ChatCompletion:
        assert model or self.model, "Model must be provided."
        while True:
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    messages=messages,
                    model=model if model else self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    seed=seed,
                    top_p=top_p,
                )
                break
            except openai.RateLimitError:
                pass
            except Exception as e:
                print(f"Error when generating: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)
        return response


def init_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    if not args.model_base or args.model_base == "None" or len(args.model_base) < 5:
        model_base = None
        model_name = "llava-v1.5-7b"
    else:
        model_base = args.model_base
        model_name = "llava-v1.5-7b-lora"
    print(f"Loading model from {model_path} with base model {model_base}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name)

    return tokenizer, model, image_processor


def read_json(file_path: str) -> list[dict] | dict:
    """
    Read JSON file, support two formats: .json and .jsonl.
    """
    ext = os.path.splitext(file_path)[-1]
    if ext == ".json" or ext == ".jsonfile":
        with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".jsonl":
        with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unspported extension {ext} for file: {file_path}")
    return data


# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
object_synonyms_txt = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, brother, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager, pedestrian
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk, coffee table
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote, remote control
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""

visual_genome_obj: list[str] = [
    "tree",
    "window",
    "shirt",
    "building",
    "person",
    "table",
    "car",
    "door",
    "light",
    "fence",
    "chair",
    "people",
    "plate",
    "glass",
    "jacket",
    "sidewalk",
    "snow",
    "flower",
    "hat",
    "bag",
    "track",
    "roof",
    "umbrella",
    "helmet",
    "plant",
    "train",
    "bench",
    "box",
    "food",
    "pillow",
    "bus",
    "bowl",
    "horse",
    "trunk",
    "clock",
    "mountain",
    "elephant",
    "giraffe",
    "banana",
    "house",
    "cabinet",
    "hill",
    "dog",
    "book",
    "bike",
    "coat",
    "glove",
    "zebra",
    "bird",
    "motorcycle",
    "lamp",
    "cow",
    "skateboard",
    "surfboard",
    "beach",
    "sheep",
    "kite",
    "cat",
    "pizza",
    "bed",
    "bear",
    "windshield",
    "towel",
    "desk",
]

# 关系的同义词，用于判断两个关系是否相等
relation_sysnonyms_txt = """
in, on, at
equals, is
belongs to, is part of
"""

coco_double_words = [
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "hot dog",
    "teddy bear",
    "home plate",
    "train track",
    "dining table",
    "coffee table",
]
animal_words = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
vehicle_words = ["jet", "train"]


def remove_negetive_sents(caption: str) -> str:
    sents: list[str] = nltk.sent_tokenize(caption)
    sents = [sent for sent in sents if "There is no" not in sent and "There are no" not in sent]
    return " ".join(sents)


def remove_woodpecker_boxes(text: str) -> str:
    """
    Remove the boxes generated by Woodpecker from the text.
    """
    text = re.sub(r"\(\[.*?\]\)", "", text)
    text = re.sub(r"\(\[.*?\]\;", "", text)
    text = re.sub(r"\[.*?\]\;", "", text)
    return text
