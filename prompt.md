我想自己做一个DPO训练集，类似generate_dataset.py中的逻辑，但不完全相同。
流程如下：
1. 读取我给定的参考数据集，内容包括 input = {image_path, question, ground_truth}
2. 对ground_truth中的物体做提取（类似generate_dataset.py中的逻辑），现在有input = {image_path, question, objects}
3. 对于每一个input，利用dino与yolo检测objects中存在的所有物体，记录它们的名称，置信度，检测盒，如果同一个物体有多个检测盒和置信度，合并成list：[object, scores, boxs]。
4. 针对每一个object，利用boxes，对输入image做mask（遮挡），把这个物体记录为masked_object, 得到遮挡之后的masked_image，将masked_image与prompt送入VLM，得到masked_response
5. 对masked_response进行解析，如果其中仍然有被遮挡的物体，记录下来这个masked_image，作为偏好对，masked_image文件名为 原图像文件名+_masked_${masked_object}+文件后缀。偏好数据集仅需记录[image_id, image_path, question, masked_image_path]


# Pipeline
1. mask必须超过一定阈值才能使用（具体根据实验）。为了mask有效且鲁棒，同一类物体只要有一个box的置信度超过了阈值，就使用这一类的所有box，保证完全遮挡物体。
2. 对于某个masked image，VLM生成10个句子，幻觉句子比例超过一定值才认为有幻觉（阈值需实验）