# Feint6K Dataset

**Feint6K** dataset for video-text understanding, from the following paper:

<div align="center">
<strong><h3><a href="https://arxiv.org/abs/2407.13094">Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data</a></h3></strong>
</div>
<div align="center">
<span><a href="https://wufeim.github.io">Wufei Ma<sup>1,2</sup></a>&nbsp;&nbsp;</span>
<span><a href="https://sites.google.com/view/kaisqu/">Kai Li<sup>1</sup></a>&nbsp;&nbsp;</span>
<span><a href="https://scholar.google.com/citations?user=h8bGMF4AAAAJ&hl=en">Zhongshi Jiang<sup>1</sup></a>&nbsp;&nbsp;</span>
<span><a href="http://www.cs.umd.edu/~mmeshry/">Moustafa Meshry<sup>1</sup></a>&nbsp;&nbsp;</span>
<span><a href="https://qihao067.github.io">Qihao Liu<sup>2</sup></a></span><br/>
<span><a href="https://csrhddlam.github.io">Huiyu Wang<sup>3</sup></a>&nbsp;&nbsp;</span>
<span><a href="https://scholar.google.com/citations?user=AliuYd0AAAAJ&hl=en">Christian Häne<sup>1</sup></a>&nbsp;&nbsp;</span>
<span><a href="https://www.cs.jhu.edu/~ayuille/">Alan Yuille<sup>2</sup></a></span>
</div>
<br/>
<div align="center">
<span><sup>1</sup>Meta Reality Labs&nbsp;&nbsp;</span>
<span><sup>2</sup>Johns Hopkins University&nbsp;&nbsp;</span>
<span><sup>3</sup>Meta AI</span>
</div>
<br/>
<div align="center">
    <span>ECCV 2024&nbsp;&nbsp;&nbsp;&nbsp;</span>
    <span><a href="https://feint6k.github.io">Project Page</a>&nbsp;&nbsp;&nbsp;&nbsp;</span>
    <span><a href="https://arxiv.org/abs/2407.13094">arXiv</a></span>
</div>

---

We propose a novel evaluation task for video-text understanding, namely <ins>*retrieval from counterfactually augmented data* (RCAD)</ins>, and a new <ins>*Feint6K*</ins> dataset, to better assess the capabilities of current video-text models and understand their limitations. To succeed on our new task, models must derive a comprehensive understanding of the video from cross-frame reasoning. Analyses show that previous video-text foundation models can be easily fooled by counterfactually augmented data and are far behind human-level performance.

From our experiments on RCAD, we identify a key limitation of current contrastive approaches on video-text data and introduce <ins>*LLM-teacher*</ins>, a more effective approach to learn action semantics by leveraging knowledge obtained from a pretrained large language model.

<p align="center">
<img src="https://github.com/user-attachments/assets/f0a3e762-f2e2-48fe-9c80-46fc998d625a" width=100% height=100% class="center">
</p>

## Data Preparation

1. Download Feint6K data (`.csv` files with counterfactually augmented captions) from [here](https://drive.google.com/drive/folders/1saIVSNPsmQZ_lmlVWzw2rkDfW67ZGAal?usp=share_link).

2. Download video data for MSR-VTT and VATEX to a video data folder, *e.g.*, `./videos`:

    ```
    ./videos
      |- msrvttvideo
      |   |- *.mp4
      |- vatexvideo
          |- *.mp4
    ```

## Example RCAD Evaluation on Feint6K Dataset

1. Compute video-text similarity matrix, *e.g.*, with [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind). Similarity matrices will be saved to `sim_mat_msrvtt.npy` and `sim_mat_vatex.npy` for RCAD on `MSR-VTT` and `VATEX` respectively.

    ```sh
    # install and activate conda environment for LanguageBind
    # see: https://github.com/PKU-YuanGroup/LanguageBind?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation
    conda activate languagebind

    python3 compute_sim_mat_languagebind.py --video_path videos
    ```

2. Compute RCAD metrics given the saved similarity matrix for any video-text model:

    ```sh
    python3 eval_rcad.py
    ```

   The RCAD results will be printed to the console, *e.g.*,

   ```
   RCAD on msrvtt: R@1=41.7 R@3=76.5 meanR=2.4 medianR=2.0
   RCAD on vatex: R@1=43.2 R@3=77.2 meanR=2.3 medianR=2.0
   ```

## Statements

All data collection and experiments in this work were conducted at JHU.

**Ethics.** We follow the ethics guidelines of ECCV 2024 and obtained Institutional Review Board (IRB) approvals prior to the start of our work. We described potential risks to the annotators, such as being exposed to inappropriate videos from public video datasets, and explained the purpose of the study and how the collected data will be used. All annotators agreed to join this project voluntarily and were paid by a fair amount as required at our institution.

## Citation

If you find this dataset helpful, please cite:

```
@inproceedings{ma2024rethinking,
  title={Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data},
  author={Ma, Wufei and Li, Kai and Jiang, Zhongshi and Meshry, Moustafa and Liu, Qihao and Wang, Huiyu and H{\"a}ne, Christian and Yuille, Alan},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```

## License
Fent6k is CC-BY-NC 4.0 licensed, as found in the [LICENSE file](https://github.com/facebookresearch/feint6k/blob/main/LICENSE).

[[Terms of Use](https://opensource.facebook.com/legal/terms)]
[[Privacy Policy](https://opensource.facebook.com/legal/privacy)]