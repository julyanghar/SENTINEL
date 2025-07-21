import copy
import glob
import json
import os
import pathlib
import random
import ssl
import sys
import time
from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from spacy import Language
from spacy.tokens import Doc
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llava.eval.utils.utils import (
    ChatCompletion,
    OpenAIModel,
    animal_words,
    coco_double_words,
    object_synonyms_txt,
    read_json,
    remove_negetive_sents,
    remove_woodpecker_boxes,
    vehicle_words,
)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

spacy.require_gpu()
gpt_model_name = "gpt-3.5-turbo"


def main(args: Namespace) -> None:
    time_start = time.time()
    gpt_model = OpenAIModel(model=gpt_model_name)
    nlp: Language = spacy.load("en_core_web_trf")

    if args.cap_folder != "":
        patterns = ["*", "*/*", "*/*/*", "*/*/*/*"]
        f_list = sum([list(glob.glob(args.cap_folder + p)) for p in patterns], [])
        cap_file_path_list: list[str] = [x for x in f_list if x.endswith(".jsonl") and args.cap_type in x]
        random.shuffle(cap_file_path_list)
        args.cap_file = cap_file_path_list[0]
    else:
        cap_file_path_list: list[str] = [args.cap_file]

    print("=======load prediction=======")
    print("load imgids file:", args.cap_file)
    _, img_ids, _ = load_generated_captions(args.cap_file, org_dir=args.org_folder)

    print("=======init evaluator=======")
    evaluator: CHAIR = CHAIR(img_ids, args.coco_path)  # 初始化 CHAIR 对象

    print("========compute=========")
    for cap_file_path in cap_file_path_list:
        print(f"Processing {cap_file_path}")
        tag: str = cap_file_path.split("/")[-1].replace(".jsonl", "")

        save_dir = pathlib.Path(cap_file_path).absolute().parent
        if (save_dir / f"{tag}_eval_Object_HalBench.json").exists():
            print("eval file already exists!")
            continue

        cap_dict = evaluator.compute_chair(cap_file_path, nlp=nlp, gpt=gpt_model, org_dir=args.org_folder)
        print("Done computing")

        save_hallucinated_words(cap_file_path, cap_dict, save_dir)
        print("Done Saving")

        print_metrics(cap_dict)

        print("Total eval time:", time.time() - time_start)


prompt_template = """You are an expert in image objects extraction according to a question answer pair. We asked an examiner to answer a question about a picture.

[Start of Question]

<image> {question}

[End of Question]

[Start of Examiner's Answer]

{answer}

[End of Examiner's Answer]


Assume that the answer is correct, please identify all visible objects that are directly shown in the image. Please following the instructions in below:

1. You should only mention objects that are explicitly mentioned in the examiner's answer.
2. You should only extract the object names without the attributes of the objects.
3. You should not include the properties of the object, like the color, material, etc. as part of the object name in your result.
4. Make your answer precise. Present the results in a JSON list format: [\"object_1\", ..., \"object_n\"].
5. You should return an empty JSON list () if no visible objects can be found.
"""


def parse_obj_list(content: str) -> list[str]:
    content: str = content.strip("```json").strip("```").strip()
    try:
        obj_list: list[str] = json.loads(content)
    except Exception:
        if '["' in content:
            try:
                obj_list = json.loads(content.strip().split("\n")[-1])
            except Exception:
                raise ValueError("Content is not json interpretable")
        else:
            raise ValueError("Content is not json interpretable")
    return obj_list


def preprocess_coh_results(caps):
    new_caps = []
    for cap in caps:
        cap_text = cap["caption"]
        if "The following is a response without hallucination." in cap_text:
            new_cap_text = cap_text.split("The following is a response without hallucination.")[-1].strip()
        elif "The following is a response with hallucination." in cap_text:
            new_cap_text = cap_text.split("The following is a response with hallucination.")[0].strip()
        elif "Generate a response without errors." in cap_text:
            new_cap_text = cap_text.split("Generate a response without errors.")[-1].strip()
        elif "Generate a response with errors." in cap_text:
            new_cap_text = cap_text.split("Generate a response with errors.")[0].strip()
        else:
            new_cap_text = cap_text
        cap["caption"] = new_cap_text
        new_caps.append(cap)

    return new_caps


lemmatizer = WordNetLemmatizer()


class CHAIR(object):
    def __init__(self, img_ids, coco_path: str):
        self.img_id_to_objects: dict[str, list[str]] = {imid: [] for imid in img_ids}
        self.coco_path: str = coco_path
        self.fail_limit = 100

        # read in synonyms
        synonyms = object_synonyms_txt.splitlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            new_synonym = [s.strip() for s in synonym]
            self.mscoco_objects.extend(new_synonym)
            for s in new_synonym:
                self.inverse_synonym_dict[s] = new_synonym[0]

        self.double_word_dict: dict[str, str] = {}  # 保存双词的映射关系
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

        self._get_annotations()

    def _load_generated_captions(self, cap_file, org_dir=None):
        """
        Meant to save time so imid_to_objects does not always need to be recomputed.
        """
        # Read in captions
        self.captions, self.img_ids, self.metrics = load_generated_captions(cap_file, org_dir=org_dir)
        for index, cap in enumerate(self.captions):
            cap["index"] = index

    def get_double_words_only(self, word_list: list[str]) -> list[str]:
        i = 0
        double_words = []
        idxs = []
        words = word_list
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                i += 1
        return double_words

    def _caption_to_words(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        """
        Input:
        一张图片的 caption

        Output:
        该 sentence 中的所有 MSCOCO 对象列表，对象对应的代表词列表，对应的索引列表，所有单词的列表

        前三个列表长度相同
        """
        # 词形还原（将单词的不同形式还原为其基本形式或词根），快一些
        # all_words = [token.lemma_ for token in nlp(caption.lower())]
        # 词语变为单数，只处理名词，慢一些
        all_words: list[str] = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(caption.lower())]

        # 将文本中的双词对象合并成一个对象
        i = 0
        double_words, idxs = [], []
        while i < len(all_words):
            idxs.append(i)
            double_word = " ".join(all_words[i : i + 2])  # 两个词组成的双词短语，用于查找映射关系
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(all_words[i])
                i += 1
        all_words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in all_words) & ("seat" in all_words):
            all_words = [word for word in all_words if word != "seat"]

        # 仅保留 caption 中的 MSCOCO 对象作为需要检测的对象
        idxs = [idxs[idx] for idx, word in enumerate(all_words) if word in set(self.mscoco_objects)]
        all_objects: list[str] = [word for word in all_words if word in set(self.mscoco_objects)]
        node_words: list[str] = []  # 代表词，即每个词的同义词的第一个词
        for obj in all_objects:
            node_words.append(self.inverse_synonym_dict[obj])
        # return all the MSCOCO objects in the caption
        return all_objects, node_words, idxs, all_words

    def _caption_objects_to_coco_objects(self, words: list[str]):
        idxs: list[int] = list(range(len(words)))
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]
        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words: list[str] = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append([word, self.inverse_synonym_dict[word]])

        # return all the MSCOCO objects in the caption
        return words, node_words, idxs

    def _get_annotations_from_segments(self):
        """
        Add objects taken from MSCOCO segmentation masks
        """

        def combine_coco_instances(annotation_path):
            if not os.path.exists(f"{annotation_path}/instances_val2014.json"):
                raise Exception("Please download MSCOCO instance annotations for val set")
            if not os.path.exists(f"{annotation_path}/instances_train2014.json"):
                raise Exception("Please download MSCOCO instance annotations for train set")

            val_instances = json.load(open(f"{annotation_path}/instances_val2014.json"))
            train_instances = json.load(open(f"{annotation_path}/instances_train2014.json"))
            all_instances = {
                "info": train_instances["info"],
                "licenses": train_instances["licenses"],
                "type": train_instances["licenses"],
                "categories": train_instances["categories"],
                "images": train_instances["images"] + val_instances["images"],
                "annotations": val_instances["annotations"] + train_instances["annotations"],
            }

            return all_instances

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments["annotations"]

        id_to_name = {}  # dict with id to synsets
        for cat in coco_segments["categories"]:
            id_to_name[cat["id"]] = cat["name"]

        for annotation in segment_annotations:
            img_id: str = annotation["image_id"]
            if img_id in self.img_id_to_objects:
                node_word = self.inverse_synonym_dict[id_to_name[annotation["category_id"]]]
                self.img_id_to_objects[img_id].append(node_word)
        print("Got annotations from segmentation masks")

    def _get_annotations_from_captions(self):
        """
        Add objects taken from MSCOCO ground truth captions
        """

        def combine_coco_captions(annotation_path):
            if not os.path.exists(f"{annotation_path}/captions_val2014.json"):
                raise Exception("Please download MSCOCO caption annotations for val set")
            if not os.path.exists(f"{annotation_path}/captions_train2014.json"):
                raise Exception("Please download MSCOCO caption annotations for train set")

            val_caps = json.load(open(f"{annotation_path}/captions_val2014.json"))
            train_caps = json.load(open(f"{annotation_path}/captions_train2014.json"))
            all_caps = {
                "info": train_caps["info"],
                "licenses": train_caps["licenses"],
                "images": val_caps["images"] + train_caps["images"],
                "annotations": val_caps["annotations"] + train_caps["annotations"],
            }

            return all_caps

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps["annotations"]

        for i, annotation in enumerate(caption_annotations):
            img_id: str = annotation["image_id"]
            if img_id in self.img_id_to_objects:
                _, node_words, _, _ = self._caption_to_words(annotation["caption"])
                self.img_id_to_objects[img_id].extend(node_words)
        print("Got annotations from ground truth captions")

    def _get_annotations(self):
        """
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        """
        self._get_annotations_from_segments()
        self._get_annotations_from_captions()

        # deduplicate
        for img_id in self.img_id_to_objects:
            self.img_id_to_objects[img_id] = set(self.img_id_to_objects[img_id])

    def _get_gpt_resp(self, gpt: OpenAIModel, data_item: dict):
        system: str = copy.deepcopy(prompt_template).format(question=data_item["question"], answer=data_item["caption"])
        messages: list[dict] = [{"role": "system", "content": system}]

        fail_cnt: int = 0
        used_tokens: dict[str, int] = {"total": 0, "input": 0, "output": 0}
        while True:
            # 处理特殊情况
            if len(data_item["caption"].strip().split()) <= 3:
                data_item["extract_objs"] = []
                print(
                    f"**[Short Answer]**@{data_item['caption']}@",
                    data_item["extract_objs"],
                )
                return data_item, used_tokens, {"total": 0, "input": 0, "output": 0}
            if fail_cnt == self.fail_limit:
                data_item["extract_objs"] = "-1\n<no_response>"
                print("**[Wrong Return]**", data_item["extract_objs"])
                return data_item, used_tokens, {"total": 0, "input": 0, "output": 0}
            resp = None
            try:
                resp: ChatCompletion = gpt.gen(
                    messages=messages,
                    return_completions=True,
                    use_parallel=False,
                    use_tqdm=False,
                )

                used_tokens["total"] += resp.usage.total_tokens
                used_tokens["input"] += resp.usage.prompt_tokens
                used_tokens["output"] += resp.usage.completion_tokens

                content: str = resp.choices[0].message.content  # '["man", "wine glass", "clock"]'
                obj_list: list[str] = parse_obj_list(content)

                # API Rest
                time.sleep(5)

                data_item["extract_objs"] = obj_list
                success_tokens = {
                    "total": resp.usage.total_tokens,
                    "input": resp.usage.prompt_tokens,
                    "output": resp.usage.completion_tokens,
                }
                return data_item, used_tokens, success_tokens
            except Exception as e:
                fail_cnt += 1
                print(f"Exception: {e}\nmessages: {messages}\nresponse: {resp}\n")
                time.sleep(5 + fail_cnt)

    def gpt_caption_processor(self, gpt: OpenAIModel, max_workers: int = 64):
        data_list: list[dict] = self.captions
        new_data = []
        all_used_tokens = {"total": 0, "input": 0, "output": 0}
        all_success_tokens = {"total": 0, "input": 0, "output": 0}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print("thread num:", len(data_list))

            futures: list[Future] = [executor.submit(self._get_gpt_resp, gpt, data_item) for data_item in data_list]

            pb = tqdm(total=len(futures))

            for completed_furure in as_completed(futures):
                pb.update(1)
                try:
                    new_data_item, used_tokens, success_tokens = completed_furure.result()  # type = List
                    all_used_tokens = {key: all_used_tokens[key] + used_tokens[key] for key in all_used_tokens.keys()}
                    all_success_tokens = {
                        key: all_success_tokens[key] + success_tokens[key] for key in all_success_tokens.keys()
                    }
                    new_data.append(new_data_item)
                except Exception as e:
                    print(f"@@@ Exception: {e}\n")
        print("Done loop, waiting resource finalization", flush=True)
        return new_data, all_used_tokens, all_success_tokens

    def postagging(self, doc: Doc) -> list[str]:
        obj_list = []
        temp_token = ""

        for token in doc:
            if token.tag_ in ["NNP", "NNPS", "NN", "NNS"]:
                temp_token += f" {token.lemma_}"
            else:
                if temp_token != "":
                    obj_list.append(temp_token.strip())
                    temp_token = ""
        if temp_token != "":
            obj_list.append(temp_token.strip())
        return obj_list

    def get_pred_objs_match(self, captions: list[dict], nlp: Language) -> list[dict]:
        new_caps = []
        all_objs: list[str] = [f"a {obj}" for caption in captions for obj in caption["extract_objs"]]
        batch_size = 50  # Adjust based on your hardware
        docs: list[Doc] = list(tqdm(nlp.pipe(all_objs, batch_size=batch_size, n_process=1), total=len(all_objs)))

        obj_index = 0  # Map the processed documents back to their original captions
        for caption in captions:
            caps_gpt_objs: list[str] = caption["extract_objs"]
            refined_objs = []

            for obj in caps_gpt_objs:
                doc: Doc = docs[obj_index]
                assert doc.text == f"a {obj}"
                obj_index += 1

                single_tokens = [token.lemma_ for token in doc]
                double_words_objects = self.get_double_words_only(single_tokens)

                if double_words_objects:
                    refined_objs += double_words_objects
                    continue

                postagging_objs: list[str] = self.postagging(doc)
                refined_objs += postagging_objs

            new_item: dict = copy.deepcopy(caption)

            # only append unique word in the list
            new_item["objs"] = list(set(refined_objs))

            new_caps.append(new_item)

        return new_caps

    def compute_chair(self, cap_file: str, nlp: Language, gpt: OpenAIModel | None = None, org_dir=None):
        """
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        """

        self._load_generated_captions(cap_file, org_dir=org_dir)

        imid_to_objects: dict[str, list[str]] = self.img_id_to_objects
        captions: list[dict] = self.captions

        if gpt is not None:
            captions, all_used_tokens, all_success_tokens = self.gpt_caption_processor(gpt=gpt)
            captions: list[dict] = self.get_pred_objs_match(captions, nlp=nlp)
        else:
            all_used_tokens = {}
            all_success_tokens = {}

        num_caps = 0.0
        num_coco_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0
        gt_word_count = 0.0
        coco_obj_cls_count = 0.0

        output = {"sentences": []}
        total_cap_word_num: int = 0
        for caption_eval in captions:
            caption: str = caption_eval["caption"]
            caption = remove_woodpecker_boxes(caption)  # Remove the boxes generated by Woodpecker from the text.
            caption = remove_negetive_sents(caption)  # Remove all the sentences with "There is no" or "There are no"
            word_len: int = len(caption.strip().split(" "))
            total_cap_word_num += word_len

            image_id: int = caption_eval["image_id"]

            # get all words in the caption, as well as corresponding node word
            if gpt is not None:
                ext_objs: list[str] = caption_eval["objs"]
                words, node_words, idxs = self._caption_objects_to_coco_objects(ext_objs)
                raw_words = ext_objs
            else:
                words, node_words, idxs, raw_words = self._caption_to_words(caption)

            gt_objects: set[str] = imid_to_objects[image_id]
            gt_word_count += len(gt_objects)
            cap_dict = {
                "image_id": caption_eval["image_id"],
                "caption": caption,  # org cap
                "mscoco_hallucinated_words": [],
                "mscoco_gt_words": list(gt_objects),  # gt coco objs
                "mscoco_generated_words": list(node_words),  # gen mapped coco objs
                "hallucination_idxs": [],
                "words": raw_words,  # gpt process -> map double words -> postagging results, or original text words lemmas
                "word_len": word_len,
            }

            cap_dict["metrics"] = {"CHAIRs": 0, "CHAIRi": 0}

            # count hallucinated words, if [word, coco_obj_cls] is unique, count as one prediction
            coco_word_count += len(node_words)
            caption_coco_obj_cls = []

            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word[-1] not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                    cap_dict["hallucination_idxs"].append(idx)
                    hallucinated = True
                else:
                    caption_coco_obj_cls.append(node_word[-1])

            caption_coco_obj_cls = set(caption_coco_obj_cls)
            # print(caption_coco_obj_cls)
            coco_obj_cls_count += len(caption_coco_obj_cls)

            # count hallucinated caps
            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1

            cap_dict["metrics"]["CHAIRs"] = int(hallucinated)
            cap_dict["metrics"]["CHAIRi"] = 0.0
            if len(words) > 0:
                num_coco_caps += 1
                cap_dict["metrics"]["CHAIRi"] = len(cap_dict["mscoco_hallucinated_words"]) / float(len(words))

            output["sentences"].append(cap_dict)

        chair_s = num_hallucinated_caps / num_caps
        chair_s_refine = num_hallucinated_caps / num_coco_caps
        chair_i = hallucinated_word_count / coco_word_count
        avg_word_len = float(total_cap_word_num) / num_caps
        obj_rec = coco_obj_cls_count / gt_word_count

        output["overall_metrics"] = {
            "CHAIRs": chair_s,
            "CHAIRs_refine": chair_s_refine,
            "CHAIRi": chair_i,
            "obj_rec": obj_rec,
            "sentence_num": num_caps,
            "coco_sentence_num": num_coco_caps,
            "coco_word_count": coco_obj_cls_count,  # predict coco object classes
            "gt_word_count": gt_word_count,  # ground truth coco object classes
            "avg_word_len": avg_word_len,
            "all_gpt_used_tokens": all_used_tokens,
            "all_gpt_success_tokens": all_success_tokens,
            "correct_rate": 1 - chair_s_refine,
            "object_correct_rate": 1 - chair_i,
        }

        return output


def load_generated_captions(cap_file, org_dir=None):
    if cap_file.endswith(".json"):
        caps = json.load(open(cap_file))
        try:
            metrics = caps["overall"]
            caps = caps["imgToEval"].values()
            img_ids = set([cap["image_id"] for cap in caps])
        except Exception:
            raise Exception(
                "Expect caption file to consist of a dictionary with sentences correspdonding to the key 'imgToEval'"
            )
    elif cap_file.endswith(".jsonl"):
        caps: list[dict] = read_json(cap_file)
        if "image_id" not in caps[0].keys():  # 检查是否需要从原始文件中读取 image_id
            try:
                assert org_dir is not None and org_dir.strip() != ""
            except Exception:
                raise Exception("Expect origin test input file directory for .jsonl cap file")
            cap_name = cap_file.split("/")[-1]
            org_name = cap_name.split("__")[0].replace("_answer", ".jsonl")
            if org_dir.endswith(".jsonl"):
                org_data_path = org_dir
            else:
                org_data_path = os.path.join(org_dir, org_name)
            org_data = read_json(org_data_path)
        metrics = {}
        new_captions: list[dict] = []
        img_ids: list[int] = []
        for i in range(len(caps)):
            if "image_id" not in caps[i].keys():
                imgid = int(org_data[i]["image_id"].strip(".jpg"))
            else:
                imgid = int(caps[i]["image_id"].strip(".jpg"))
            img_ids.append(imgid)
            if "prompt" in caps[i].keys():
                question = caps[i]["prompt"]
            elif "question" in caps[i].keys():
                question = caps[i]["question"]
            else:
                question = "Describe this image."

            if "text" in caps[i].keys():
                answer = caps[i]["text"].replace("Assistant:", "").strip()
            elif "answer" in caps[i].keys():
                answer = caps[i]["answer"].replace("Assistant:", "").strip()
            elif "caption" in caps[i].keys():
                answer = caps[i]["caption"].replace("Assistant:", "").strip()
            else:
                raise Exception("Expect 'answer' or 'text' in generated file")
            new_captions.append({"image_id": imgid, "question": question, "caption": answer})
        caps = new_captions
        img_ids = set(img_ids)
    elif "." not in cap_file:
        caps = json.load(open(cap_file))
        try:
            assert "raw_question" in caps[0].keys()
        except Exception:
            raise Exception("Expect origin test input file directory for .jsonl cap file")
        img_ids = set([int(cap["question_id"].replace(".jpg")) for cap in caps])
        metrics = {}
        new_captions = []
        for item in caps:
            new_item = {
                "image_id": int(item["question_id"].replace(".jpg", "")),
                "question": item["raw_question"],
                "caption": item["answer"].replace("Assistant:", "").strip(),
            }
            new_captions.append(new_item)
        caps = new_captions
    if "coh" in cap_file:
        caps = preprocess_coh_results(caps)

    return caps, img_ids, metrics


def save_hallucinated_words(cap_file: str, cap_dict: dict, save_dir: str) -> None:
    tag: str = cap_file.split("/")[-1].replace(".jsonl", "")
    sentences: list[dict] = cap_dict["sentences"]
    overall: dict = cap_dict["overall_metrics"]
    with open(os.path.join(save_dir, f"{tag}_eval_Object_HalBench.json"), "w") as f:
        json.dump(sentences, f, indent=2)
    with open(os.path.join(save_dir, f"{tag}_eval_Object_HalBench_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)


def print_metrics(hallucination_cap_dict) -> None:
    sentence_metrics = hallucination_cap_dict["overall_metrics"]
    metric_string = "%0.001f\t%0.001f\t%0.001f\t%d\t%d\t%0.01f" % (
        sentence_metrics["CHAIRs"] * 100,
        sentence_metrics["CHAIRs_refine"] * 100,
        sentence_metrics["CHAIRi"] * 100,
        sentence_metrics["sentence_num"],
        sentence_metrics["coco_sentence_num"],
        sentence_metrics["avg_word_len"],
    )

    print("CHAIRs\tCHAIRsr\tCHAIRi\tsent_num\tcoco_num\tavg_len")
    print(metric_string)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--cap_file", type=str, default="")
    parser.add_argument("--cap_folder", type=str, default="")
    parser.add_argument("--org_folder", type=str, default="")
    parser.add_argument("--cap_type", type=str, default="")
    parser.add_argument("--coco_path", type=str, default="./coco2014/annotations")
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--openai_key", type=str, default="")
    args: Namespace = parser.parse_args()

    return args


if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args)
