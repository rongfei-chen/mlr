# mlr
Multimodal Language Representation learning

This repository provides pretrained models that produce Multimodal Language embeddings to be used in several Multimodal Language Analysis tasks such as Emotion Recognition, Sentiment Analysis, Persuasiveness Prediction etc.

For a complete guide and performance metrics of the methods used in this repository, we refer the user to our recent work [Unsupervised Multimodal Language Representations using Convolutional Autoencoders](https://arxiv.org/abs/2110.03007)

## Pretrained models

A range of different pretrained models is provided:
- directory pretrained_models/ConvAE/different_pretraining contains models that are trained on all possible combinations on the CMU-MOSEI, CMU-MOSI and IEMOCAP datasets for all modalities
- directory pretrained_models/ConvAE/different_modalities contains models that are trained on CMU-MOSEI and IEMOCAP for all possible combinations audio, vision and text modalities

## Dataloading
Dataloading is done using the [MultimodalSDK_loader](https://github.com/lobracost/MultimodalSDK_loader) repository that performs feature extraction using the well-established implementation of [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK).


## Get representations

In order to get representations for a downstream task, run (example is on cmumosei - you can create your own dataloader for other datasets based on utils.datalodaing.classification_dataloaders function):

```
import utils.dataloading as dataloading
import utils.architectures as architectures

device = "cpu"
model_name = "pretrained_models/ConvAE/different_pretraining/ConvAE_cmumosei_iemocap.pt"
model = architectures.ConvAutoEncoder(20, 409)
model.load_state_dict(torch.load(model_name))

dataset_name = "cmumosei"
dataset = dataloading.dataset_features(dataset_name)

train_loader, val_loader, test_loader, _ = dataloading.classification_dataloaders(dataset_name, dataset, 0)
x_train, y_train = get_representations(model, train_loader, device)
```

## Train new representations
In order to train representations just run from bash:
```
 python3 utils/train.py
```

In case you want to include new datasets (that have to be processed by CMU-MultimodalSDK) in the representation learning procedures, you have to firstly include your datasets in the utils.dataloading.representation_dataloaders function. 

## Dowstream Classification

In order to test the perfomance of a Logistic Regression algorithm  in the downstream tasks of Emotion Recognition (IEMOCAP) and Sentiment Analysis (CMU-MOSEI) with the use of the trained representations, run:

```
python3 downstream_classification.py
```

### Citation
```
@article{koromilas2021unsupervised,
  title={Unsupervised Multimodal Language Representations using Convolutional Autoencoders},
  author={Koromilas, Panagiotis and Giannakopoulos, Theodoros},
  journal={arXiv preprint arXiv:2110.03007},
  year={2021}
}
```
