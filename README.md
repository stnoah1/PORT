# Pose Relation Transformer (PORT)
Official PyTorch implementation of **Pose Relation Transformer: Refine Occlusions for Human Pose Estimation**. 

The `model.py` will be available soon!

## Abstract
<img width="1025" alt="qualitative" src="https://user-images.githubusercontent.com/37060326/194792799-9315b317-b0b3-4b99-9105-08d96727993a.png">


Accurately estimating the human pose is an essential task for many applications in robotics. However, existing
pose estimation methods suffer from poor performance when occlusion occurs. Recent advances in NLP have been very
successful in predicting the missing words conditioned on visible words. We draw upon the sentence completion analogy in NLP
to guide our model to address occlusions in the pose estimation problem. We propose a novel approach that can mitigate the
effect of occlusions that is motivated by the entence completion task of NLP. In an analogous manner, we designed our model
to reconstruct occluded joints given the visible joints utilizing joint correlations by capturing the implicit joint occlusions.
Our proposed POse Relation Transformer (PORT) captures the global context of the pose using self-attention and a local context
by aggregating adjacent joint features. To train PORT to learn joint correlations, we guide PORT to reconstruct randomly
masked joints, which we call Masked Joint Modeling (MJM). PORT trained with MJM adds to existing keypoint detection
methods and successfully refines occlusions. Notably, PORT is a model-agnostic plug-in for pose refinement under occlusion that
can be plugged into any keypoint detector with substantially low computational costs. We conducted extensive experiments
to demonstrate that PORT mitigates the occlusion effects on the hand and body pose estimation. Strikingly, PORT improves the
pose estimation accuracy of existing human pose estimation methods up to 16% with only 5% of additional parameters.

## Dependencies

- Python >= 3.6
- PyTorch >= 1.7.0
- NVIDIA Apex
- tqdm

## Data Preparation
Ideally, PORT can be plugged into any detection-based pose estimation backbone.
We recommend you to apply PORT to the result files of backbones which only contain joint locations.
[MMPose](https://mmpose.readthedocs.io/en/latest/) provides pre-trained backbones and pipelines to make the inference result files.
You may want to follow the [instruction of MMPose](https://mmpose.readthedocs.io/en/latest/get_started.html#inference-with-pre-trained-models).

After you get the result files of backbones, you have to locate it into `data/` folder as well as annotation files.

For example, if you want to refine the result of `mobilenetv2` on `CMU panoptic hand` dataset, the directory should be like the following.

```
- data/
    - panoptic/
        - annotations/
            - panoptic_test.json
            - panoptic_train.json
        - mobilenetv2_panoptic_256x256_test.pkl
```
Our repository already contains the annotations and mobilenetv2 results on panoptic dataset.

## Training and Testing
- We set the seed number for Numpy and PyTorch as 0 for reproducibility.
- If you want to change the masking proportion, change `mask_prob` into other value.

This is an example command for training PORT on Panoptic dataset, and refining the result of `mobilenetv2` backbone.
```
python train.py --batch_size=128 --dataset=panoptic --epochs=100 --lr=0.001 --mask_prob=0.4 \
    --random_seed=0 --thd_percent_st=0 --thd_percent_offset=40 --thd_percent_stride=2 --proj_kernels 1 3\
    --kpt2d_test_pred_path=data/panoptic/mobilenetv2_panoptic_256x256_test.pkl\
    --train_ann_path=data/panoptic/annotations/panoptic_train.json\
    --test_ann_path=data/panoptic/annotations/panoptic_test.json
```

## Acknowledgements
We utilized the pre-trained backbones from [MMPose](https://mmpose.readthedocs.io/en/latest/), and our architecture code is based on [CvT](https://github.com/microsoft/CvT).

Thanks to the original authors for their works!
