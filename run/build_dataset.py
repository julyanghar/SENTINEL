import random
from time import time

from model.auxiliary.dataset import DataPoint
from model.auxiliary.datastate import DataStateForBuildDataset
from model.auxiliary.global_vars import GVars
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel
from model.others.sg_parser import SGParser
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel
from model.utils.gen_utils import GenOutput, get_generator
from run.utils import (  # noqa
    b_get_hallu_objects,
    extract_obj_from_textgraphs,
    extract_obj_w_gt,
    get_finish_flag,
    log_progress,
    object_in_set,
    objects_in_set,
    refModel,
    resolve_corefs,
    save_result,
    yolo_detect,
)

DEBUG = True  # 调试模式
HALLUCI_CONTEXT = False  # 是否使用含幻觉的句子添加到 context


def save_data_state(
    res_save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel | None = None,
    wn: WordnetModel | None = None,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> None:
    # For save relevant info about the data
    s.hard_positive = [
        obj for obj in s.yolo_result.labels if not object_in_set(obj, set(s.flat_gen_objs), spacy, wn, inv_synonym_map)
    ]
    s.small_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        and s.yolo_result.get_largest(obj)
        and (s.yolo_result.get_largest(obj)["xywhn"][2] * s.yolo_result.get_largest(obj)["xywhn"][3] < 0.02)
    ]
    s.edge_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        if s.yolo_result.get_farthest_to_edge(obj)
        and (
            min(
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],
            )
            < 0.1  # 如果最小距离小于 0.1，则认为物体接近边缘
        )
    ]

    save_result(
        res_save_path,
        {
            "image_id": s.data.image_id,
            "image_path": s.data.image_path,
            "question": s.question,
            "caption": s.assistant,
            "sentences_cnt": s.gen_sents_cnt,
            "hallu_objects": s.hallu_objects,
            "uncertain_objects": s.uncertain_objects,
            "nonhallu_objects": s.nonhallu_objects,
            "hard_positive": s.hard_positive,
            "small_objects": s.small_objects,
            "edge_objects": s.edge_objects,
        },
    )


def maybe_build_pair(
    save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> int:
    """Build data pair and return the index of the best sentence"""

    def create_pairs(win_candidates: list[tuple[int, list[str]]], lose_candidates, pair_type: str) -> list[dict]:
        return [
            {
                "image_id": s.data.image_id,
                "image_path": s.data.image_path,
                "question": s.data.question,
                "context": s.assistant,
                "y_win": new_sentences[win_idx],
                "y_lose": new_sentences[lose_idx],
                # Additional information
                "nonhallu_objects": s.nonhallu_objects,
                "context_gen_objects": s.context_gen_objects,
                "context_gen_hallu_objects": s.context_gen_hallu_objects,
                "objects_of_y_win": objects,
                "hallu_objects_of_y_lose": hallu_objects,
                "is_last_sent": s.is_finished,
                "type": pair_type,
            }
            for (win_idx, objects), (lose_idx, hallu_objects) in zip(win_candidates, lose_candidates)
        ]

    new_sentences: list[str] = s.generated_sentences[-1]
    if len(new_sentences) <= 1:
        return 0

    step_idx = s.now_step_idx
    objects_list, nonhallu_objects_list, hallu_objects_list = (  # noqa
        s.gen_objs(step_idx),
        s.nonhallu_objs(step_idx),
        s.hallu_objs(step_idx),
    )

    nonhallu_candidates: list = [
        (i, objects)
        for i, (objects, hallu_objects) in enumerate(zip(objects_list, hallu_objects_list))
        if len(objects) >= 1
        and not hallu_objects
        and not objects_in_set(objects, s.uncertain_objects, spacy, wn, inv_synonym_map, check_type="any")
    ]
    hallu_candidates: list = [
        (i, hallu_objects) for i, hallu_objects in enumerate(hallu_objects_list) if len(hallu_objects) >= 1
    ]

    success_explore_candidates, normal_nonhallu_candidates = [], []
    for idx, objects in nonhallu_candidates:
        if not objects_in_set(objects, s.context_gen_objects, spacy, wn, inv_synonym_map, check_type="all"):
            success_explore_candidates.append((idx, objects))
        else:
            normal_nonhallu_candidates.append((idx, objects))

    num_pairs = min(len(normal_nonhallu_candidates), len(hallu_candidates))

    all_results_list = create_pairs(normal_nonhallu_candidates[:num_pairs], hallu_candidates[:num_pairs], "y+")

    save_result(save_path.replace(".jsonl", "_data_pair.jsonl"), all_results_list)

    if HALLUCI_CONTEXT:
        if hallu_candidates:
            return random.choice([idx for idx, _ in hallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))
    else:
        if success_explore_candidates:
            return random.choice([idx for idx, _ in success_explore_candidates])
        elif normal_nonhallu_candidates:
            return random.choice([i for i, _ in normal_nonhallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))


def run_build_dataset(datalist: list[DataPoint], batch_size: int) -> None:
    logger, save_path, model_dir, alter_device = GVars.logger, GVars.save_path, GVars.model_dir, GVars.alter_device
    generator = get_generator(use_vllm=True, debug=DEBUG)

    # Object detectors
    DINO_detector = DINO("base", model_dir=model_dir, device=alter_device, logger=logger)
    yolo = YoloModel("yolo11x", model_dir=model_dir, logger=logger)  # "yolov8x-worldv2"

    # NLP tools
    SG_parser = SGParser(DEBUG, "base", model_dir, device=alter_device, logger=logger)
    spacy = SpacyModel(model_size="md", model_dir=model_dir, device=alter_device, logger=logger)
    wn = WordnetModel(logger=logger)
    ref = refModel(args=GVars.args)

    data_states: list[DataStateForBuildDataset] = []  # 正在处理的数据点，个数为 batch_size
    num_of_data, finished_data_num = len(datalist), 0

    logger.info(f"Start processing {num_of_data} data points.")

    while len(datalist) > 0 or len(data_states) > 0:
        start_time = time()

        # 装载 datalist 当中，个数为 batch_size 的数据点
        while len(data_states) < batch_size and len(datalist) > 0:
            tmp_data = datalist.pop(0)
            data_states.append(DataStateForBuildDataset(data=tmp_data))

        yolo_detect(yolo, data_states)

        out: GenOutput = generator.gen(
            images=[s.image for s in data_states],
            users=[s.question for s in data_states],
            assistants=[s.assistant for s in data_states],
            do_sample=True,
            n=10,
            temp=0.7,
            force_list=True,
            single_sentence=True,
        )
        b_new_sents: list[list[str]] = out.outputs

        for idx, (new_sents, s) in enumerate(zip(b_new_sents, data_states)):
            b_new_sents[idx], s.is_finished = get_finish_flag(new_sents, remove_duplicates=True)
            del idx, new_sents

        context = [s.assistant for s in data_states]
        b_resolved_new_sents: list[list[str]] = resolve_corefs(spacy, b_new_sents, context, 1)
        del context

        b_object_lists: list[list[list[str]]] = []
        for s, new_sents in zip(data_states, b_resolved_new_sents):
            object_lists: list[list[str]] = extract_obj_w_gt(
                new_sents,
                ref.valid_nouns,
                ref.double_words,
                ref.inv_syn_map,
                wn,
                force_list=True,
                return_repr=False,
            )

            textgraphs: list[list[list[str]]] = SG_parser.pharse(new_sents, force_list=True)
            new_object_lists: list[list[str]] = extract_obj_from_textgraphs(textgraphs, spacy, wn, force_list=True)
            object_lists = [objects + new_objects for objects, new_objects in zip(object_lists, new_object_lists)]

            del textgraphs
            b_object_lists.append(object_lists)

        b_haluci_objects_list, b_nonhallu_objects_list = b_get_hallu_objects(
            b_object_lists,
            [s.nonhallu_objects for s in data_states],
            [s.hallu_objects for s in data_states],
            spacy=spacy,
            wn=wn,
            images=[s.image for s in data_states],
            dino=DINO_detector,
            b_yolo_results=[s.yolo_result.labels for s in data_states] if yolo else None,
            yolo_labels=yolo.labels if yolo else None,
            b_uncertain_objects=[s.uncertain_objects for s in data_states],
            b_detector_rejects=[s.detector_reject for s in data_states],
            inv_syn_map=ref.inv_syn_map,
        )

        for s, new_sents, object_lists, haluci_objects_list, nonhallu_objects_list in zip(
            data_states, b_resolved_new_sents, b_object_lists, b_haluci_objects_list, b_nonhallu_objects_list
        ):
            if not new_sents:
                continue
            s.generated_sentences.append(new_sents)

            # list 未去重
            s.generated_objects.append(object_lists)
            s.generated_hallu_objects.append(haluci_objects_list)
            s.generated_nonhallu_objects.append(nonhallu_objects_list)

            best_idx: int = maybe_build_pair(save_path, s, spacy, wn, ref.inv_syn_map)
            s.app_assistant(new_sents, best_idx)

        [save_data_state(save_path, s, spacy, wn, ref.inv_syn_map) for s in data_states if s.is_finished]

        finished_data_num += len([s for s in data_states if s.is_finished])
        log_progress(logger, finished_data_num, num_of_data, batch_size, time() - start_time)
        data_states = [s for s in data_states if not s.is_finished]
