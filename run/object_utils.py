import json
import os
import re

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
object_synonyms_txt = """
person, girl, boy, man, woman, kid, child, chef, baker, people, rider, children, worker, sister, brother, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager, pedestrian
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
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
sports ball
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
tv, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote control
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
    import nltk

    sents: list[str] = nltk.sent_tokenize(caption)
    sents = [sent for sent in sents if "There is no" not in sent and "There are no" not in sent]
    return " ".join(sents)


def save_result(save_path: str, results: dict | list[dict]):
    if not save_path or not results:
        return
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "a+", encoding="utf-8") as f:
        if isinstance(results, list):
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(results, ensure_ascii=False) + "\n")


def read_json(file_path: str) -> list[dict] | dict:
    from model.utils.utils import read_json

    return read_json(file_path)


def parse_synonyms(synonym_text: str) -> tuple[list[str], dict[str, str]]:
    """
    解析同义词文本，返回所有对象及其对应的代表词（即每个同义词映射到其对应的代表词）。

    参数:
    - synonym_text: 包含同义词的文本，每行一个同义词组，逗号分隔。

    返回:
    - 所有对象的列表。
    - 每个同义词映射到其对应的代表词的字典。
    """
    synonym_lines: list[str] = synonym_text.splitlines()
    synonyms: list[list[str]] = [s.strip().split(", ") for s in synonym_lines]
    objects: list[str] = []  # 所有对象
    inverse_synonym_mapping = {}  # 将每个同义词映射到其对应的代表词（即第一个同义词）
    for synonym in synonyms:
        synonym: list[str] = [s.strip() for s in synonym if s]
        objects.extend(synonym)
        for s in synonym:
            inverse_synonym_mapping[s] = synonym[0]

    return objects, inverse_synonym_mapping


def get_object_n_represent() -> tuple[list[str], dict[str, str]]:
    """返回 COCO 数据集中的所有对象及其对应的代表词（即每个同义词映射到其对应的代表词）"""
    return parse_synonyms(object_synonyms_txt)


def get_vg_obj() -> list[str]:
    """返回 Visual Genome 数据集中的所有对象"""
    return visual_genome_obj


def remove_woodpecker_boxes(text: str) -> str:
    """
    Remove the boxes generated by Woodpecker from the text.
    """
    text = re.sub(r"\(\[.*?\]\)", "", text)
    text = re.sub(r"\(\[.*?\]\;", "", text)
    text = re.sub(r"\[.*?\]\;", "", text)
    return text


def get_double_word_dict() -> dict[str, str]:
    double_word_dict: dict[str, str] = {}  # 保存双词的映射关系
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for vehicle_word in vehicle_words:
        double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
    double_word_dict["bow tie"] = "tie"
    double_word_dict["toilet seat"] = "toilet"
    double_word_dict["wine glas"] = "wine glass"
    return double_word_dict


if __name__ == "__main__":
    # mscoco_objects, inverse_synonym_mapping = get_object_n_represent()
    # print(mscoco_objects)
    # print(inverse_synonym_mapping)
    # print(get_relation_synonyms())
    print(get_double_word_dict())
