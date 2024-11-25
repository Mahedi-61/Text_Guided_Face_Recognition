# Text-Guided Face Recognition using Multi-Granularity Cross-Modal Contrastive Learning (WACV 2024)

[https://openaccess.thecvf.com/content/WACV2024/papers/Hasan_Text-Guided_Face_Recognition_Using_Multi-Granularity_Cross-Modal_Contrastive_Learning_WACV_2024_paper.pdf](paper), [http://arxiv.org/abs/2312.09367](arxiv)

<img src="tgfr.png" width="600"> 

## Introduction
We introduce text-guided face recognition (TGFR) to analyze the impact of integrating facial attributes in the form of natural language descriptions. We hypothesize that adding semantic information into the loop can significantly improve the image understanding capability of an FR algorithm compared to other soft biometrics. However, learning a discriminative joint embedding within the multimodal space poses a considerable challenge due to the semantic gap in the unaligned image-text representations, along with the complexities arising from ambiguous and incoherent textual descriptions of the face. To address these challenges, we introduce a face-caption alignment module (FCAM), which incorporates cross-modal contrastive losses across multiple granularities to maximize the mutual information between local and global features of the face-caption pair. Within FCAM, we refine both facial and textual features for learning aligned and discriminative features. We also design a face-caption fusion module (FCFM) that applies fine-grained interactions and coarse-grained associations among cross-modal features. 


## Update
* [2024.11.25] Paper and README.md updated
* [2024.11.20]: Code Released !


## Pre-trained models
* Will be provided later


## Requirements
* [PyTorch](https://pytorch.org/) version >= 2.5.1

* Install other libraries via
```
pip install -r requirements.txt
```


## Training
### Datasets
* <span id="head-mmdata"> **Multi-Modal-CelebA-HQ** </span>

  Multi-Modal-CelebA-HQ is a large-scale face image dataset for text-to-image-generation, text-guided image manipulation, sketch-to-image generation, GANs for face generation and editing, image caption, and VQA.
  * **For information:**  ⇒ [[Paper](https://arxiv.org/pdf/2012.03308.pdf)] [[Website](https://github.com/weihaox/Multi-Modal-CelebA-HQ-Dataset)] [[Download](https://drive.google.com/drive/folders/1eVrGKfkbw7bh9xPcX8HJa-qWQTD9aWvf)]
    * Number of images (from Celeba-HQ): 30,000 (**Training**: 24,000. **Testing**: 6,000.)
    * Descriptions per image: 10 Captions


* <span id="head-celebad"> **CelebA-Dialog** </span>

  CelebA-Dialog is a large-scale visual-language face dataset.
  * **For information:**  ⇒ [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Talk-To-Edit_Fine-Grained_Facial_Editing_via_Dialog_ICCV_2021_paper.pdf)] [[Website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)] [[Download](https://github.com/yumingj/Talk-to-Edit)]
    * Number of identities: 10,177
    * Number of images: 202,599 
    * 5 fine-grained attributes annotations per image: Bangs, Eyeglasses, Beard, Smiling, and Age

* <span id="head-celebad"> **Face2Text** </span>

  Face2Text is a small-scale visual-language face dataset.
  * **For information:**  ⇒ [[Paper](https://arxiv.org/abs/2205.12342)] [[Download](https://github.com/mtanti/face2text-dataset)]
    * Number of identities: 6,193
    * Number of images: 10,559 
  

### BERT-based text encoders
1. Download the datasets from their original source.
2. Download the pre-trained weights of the image encoders in the ```weights/pretrained``` directory.
3. In ```cfg/train_bert.yml```, set the paths for the data, weights, and checkpoints.
4. Pre-trained the TGFR using atleast 2 RTX 6000 GPUs for BERT text encoder:
<pre>python3 src/train_encoders_bert.py</pre> 
5. Train the fusion layer by setting the ```cfg/fusion_bert.yml```
<pre>python3 src/fusion_bert.py</pre>                                             
6. Evaluate the results.
<pre>python3 src/test.py</pre>        
                                                                                          

### LSTM-based text encoders
1. Download the datasets from their original source.
2. Download the pre-trained weights of the image encoders in the ```weights/pretrained``` directory.
3. In ```cfg/train_lstm.yml```, set the paths for the data, weights, and checkpoints.
4. Pre-trained the TGFR using for LSTM text encoder:
<pre>python3 src/train_encoders_lstm.py</pre> 
5. Train the fusion layer by setting the ```cfg/lstm_bert.yml```
<pre>python3 src/lstm_bert.py</pre>                                             
6. Evaluate the results.
<pre>python3 src/test.py</pre>  


## Citation
If you use our work, please cite:
```
@InProceedings{Hasan_2024_WACV,
    author    = {Hasan, Md Mahedi and Sami, Shoaib Meraj and Nasrabadi, Nasser},
    title     = {Text-Guided Face Recognition Using Multi-Granularity Cross-Modal Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5784-5793}
}

@InProceedings{Hasan_2023_IJCB,
    author={Hasan, Md Mahedi and Nasrabadi, Nasser},
    booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)}, 
    title={Improving Face Recognition from Caption Supervision with Multi-Granular Contextual Feature Aggregation}, 
    year={2023},
    pages={1-10}
}
```

## Acknowledgement
This code borrows heavily from [AttnGAN](https://github.com/taoxugit/AttnGAN) repository. Many thanks

