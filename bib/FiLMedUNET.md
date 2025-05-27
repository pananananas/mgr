Proceedings of Machine Learning Research 143:416–430, 2021

MIDL 2021 – Full paper track

Beneﬁts of Linear Conditioning with Metadata for Image
Segmentation

andreanne.lemay@polymtl.ca

Andreanne Lemay 1,2
1 NeuroPoly Lab, Institute of Biomedical Engineering, Polytechnique Montreal, Canada
2 Mila, Quebec AI Institute, Canada
Charley Gros1,2
Olivier Vincent1,2
Yaou Liu3
3 Beijing Tiantan Hospital, Capital Medical University, China
Joseph Paul Cohen∗2,4
4 Stanford University Center for Artiﬁcial Intelligence in Medicine & Imaging
Julien Cohen-Adad∗1,2,5
5 Functional Neuroimaging Unit, CRIUGM, University of Montreal, Montreal, Canada

charley.gros@gmail.com
ovincent.poly@gmail.com
yaouliu80@163.com

joseph@josephpcohen.com

jcohen@polymtl.ca

Abstract
Medical images are often accompanied by metadata describing the image (vendor, acquisi-
tion parameters) and the patient (disease type or severity, demographics, genomics). This
metadata is usually disregarded by image segmentation methods. In this work, we adapt a
linear conditioning method called FiLM (Feature-wise Linear Modulation) for image seg-
mentation tasks. This FiLM adaptation enables integrating metadata into segmentation
models for better performance. We observed an average Dice score increase of 5.1% on
spinal cord tumor segmentation when incorporating the tumor type with FiLM. The meta-
data modulates the segmentation process through low-cost aﬃne transformations applied
on feature maps which can be included in any neural network’s architecture. Addition-
ally, we assess the relevance of segmentation FiLM layers for tackling common challenges
in medical imaging: multi-class training with missing segmentations, model adaptation to
multiple tasks, and training with a limited or unbalanced number of annotated data. Our
results demonstrated the following beneﬁts of FiLM for segmentation: FiLMed U-Net was
robust to missing labels and reached higher Dice scores with few labels (up to 16.7%) com-
pared to single-task U-Net. The code is open-source and available at www.ivadomed.org.
Keywords: Deep learning, linear conditioning, segmentation, metadata, task adaptation.

1. Introduction

Segmentation tasks in the medical domain are often associated with metadata: medical
condition of the patients, demographic speciﬁcations, acquisition center, acquisition pa-
rameters, etc. Depending on which structure is segmented, these metadata can help deep
learning models improve their performance, however, metadata is usually overlooked. In this
work, we improve segmentation models using recent advances in visual question answering
called FiLM (Perez et al., 2018; de Vries et al., 2017) (Feature-wise Linear Modulation).
Using FiLM to condition a segmentation model enables the integration of prior metadata
into neural networks through linear modulation layers. For instance, knowledge of the

∗ Contributed equally

© 2021 A.L. , C. Gros, O. Vincent, Y. Liu, J.P. Cohen & J. Cohen-Adad.

Benefits of Linear Conditioning with Metadata for Image Segmentation

tumor type could provide useful information to the model. (Rebsamen et al., 2019) demon-
strated that by stratifying the learning by brain tumor type, high-grade glioma, or low-grade
glioma, segmentation could be improved. With FiLM, the tumor type information can be
included without requiring multiple models as done in (Rebsamen et al., 2019). The input
metadata generates feature-speciﬁc aﬃne coeﬃcients learned during training, enabling the
model to modulate the segmentation output to improve its performance.

The metadata could also be exploited for task adaptation. When training a multi-class
segmentation model, each class needs to be annotated on every image, as missing labels will
hamper the learning (Zhou et al., 2019). Label availability often represents a bottleneck in
deep learning (Minaee et al., 2020). Segmentation is costly in terms of time, money, and
logistics (Bhalgat et al., 2018). For instance, chest CT scans contain hundreds of 2D scans
(up to 861 axial slices in the dataset used for this work) depending on the resolution. As a
reference, Google sets the price of image segmentation to 870 USD for 1000 images 1, which
totals 435 USD for a single subject with 500 axial slices. For medical segmentation requiring
expert knowledge (e.g., tumor segmentation), this price could be higher considering the
hourly wage of a radiologist. As for the time, (Ciga and Martel, 2021) reports that it takes
between 15 minutes and two hours depending on the size and resolution to segment a single
image of lymph nodes for breast cancer. An approach dealing with missing modalities and
requiring fewer labels can reduce the monetary and time-related costs.

We hypothesize that conditioning the model based on the organ to be segmented (e.g.,
“kidney”, “liver”) will make it robust to missing segmentations. A multi-class model could
then be trained on data from multiple datasets with a single class annotated in each. Since
the diﬀerent tasks share weights, fewer labels are required for a given class as the model can
learn from the other tasks. This enables the model to easily adapt a single segmentation
model to several tasks requiring only a small amount of annotations for novel tasks.

1.1. Prior work

Conditional linear modulation was introduced in many deep learning ﬁelds: visual rea-
soning (Perez et al., 2018; de Vries et al., 2017), style transfer (Dumoulin et al., 2017),
speech recognition (Kim et al., 2017), domain adaptation (Li et al., 2018), few-shot learn-
ing (Oreshkin et al., 2018), to name a few. In the medical image ﬁeld, FiLM was leveraged
for learning when limited or no annotation is available for one modality (Chartsias et al.,
2020). Image reconstruction was performed with FiLM to enable self-supervised learning
of the anatomical and modality factors of an image. Modality factors were passed through
FiLM to modulate anatomical factors generating a reconstructed image of a given modality.
While in (Chartsias et al., 2020) information extracted from the image is used for modula-
tion, in this work, we want to assess the impact of integrating metadata that is not directly
encoded in the image.

The adaptation of FiLM (i.e., linear conditioning) for segmentation was experimented on
cardiovascular magnetic resonance modulated by the distribution of class labels (Jacenk´ow
et al., 2019), on ACDC with modulation on spatio-temporal information (Jacenk´ow et al.,
2020) and on multiple sclerosis lesions with a FiLMed U-Net conditioned on the modality
(T2-weighted or T2star-weighted) (Vincent et al., 2020). (Jacenk´ow et al., 2019) had con-

1. https://cloud.google.com/ai-platform/data-labeling/pricing

417

Benefits of Linear Conditioning with Metadata for Image Segmentation

sistent improvement by including the prior information on an encoder-decoder architecture
but mitigated results on the U-Net architecture. Results from (Vincent et al., 2020) were
inconclusive regarding the performance of FiLM compared to a regular U-Net. A possible
explanation for this lack of improvement is that the modality-related features might already
be encoded in the regular U-Net, therefore the metadata added to FiLM is not informative
enough and thus does not translate to an increase in segmentation performance. In light of
these results, in the present work, we generalized the modiﬁed-FiLM implementation to be
able to modulate a model by inputting any type of discrete metadata data.

1.2. Contribution

The key contributions of this work are: (i) We introduce an adaptation of linear conditioning
(Perez et al., 2018) based on metadata for segmentation tasks using the U-Net architecture.
(ii) We demonstrate that including metadata can contribute to the model’s performance.
As a proof of concept, we input the spinal cord tumor type (astrocytoma, ependymoma,
hemangioblastoma), which is often associated with its size, composition, and anatomical
location. The tumor type knowledge led to an average Dice score improvement of 5.1%.
(iii) We show that robust learning with missing annotations can be achieved with FiLM.
Moreover, we illustrate that linear modulation enables task adaptation with fewer labeled
data when jointly trained on multiple tasks. A Dice score improvement of up to 16.7% was
observed when using our approach with a limited number of annotations compared to a
single class U-Net.

2. Methods

2.1. Architecture and Implementation

The core architecture is based on the 2D U-Net (Ronneberger et al., 2015) (Figure 1). The
model has two inputs: the image and the one-hot encoded metadata (i.e., prior knowledge).
FiLM layers and generator are responsible for conditioning the neural network with the
given metadata. Two parameters, γ(i) and β(i), are required to linearly modulate the inputs
of the ith FiLM layer. The metadata is passed through a multi-layer perceptron (i.e., FiLM
generator) with two hidden layers (64 and 16 neurons). The FiLM generator outputs one
value of γ and β for each ﬁlter (i.e., feature extractor) which are respectively multiplied
and added by the FiLM layers to each convolutional feature map. The computational cost
of FiLM is low and independent of the image resolution. The weights from the generator
are shared for a more eﬃcient learning (Perez et al., 2018). Since the input of the FiLM
generator is the same, the same features should be extracted from the metadata. The values
are constrained between 0 and 1 due to the sigmoid activation. Preliminary experiments
favored sigmoid over ReLU or tanh activation function for the FiLM parameters. γ(i)
values near 0 silence some features, while γ(i) values near 1 output the key features. Since
the linear modulation is computationally inexpensive, FiLM layers were placed after each
convolutional unit to ensure the metadata is properly used by the network. The code is
open-source and available in the ivadomed toolbox (Gros et al., 2021).

418

Benefits of Linear Conditioning with Metadata for Image Segmentation

Figure 1: FiLMed U-Net architecture of depth 3. Depth describes the number of maximum pooling
or up convolutions in the U-Net. γ and β values are generated using a multi-layer perceptron with
shared weights across FiLM layers. γ and β have the same shape as the input. An element-wise
multiplication is applied between the input and γ while the β is added.

2.2. Experiment 1: Segmentation using relevant metadata

This experiment assessed the relevance of including metadata during the training.

2.2.1. Dataset: Spinal cord tumor

We used a spinal cord tumor segmentation dataset (Lemay et al., 2021). The dataset
included 343 MRI scans, where each image was associated with one of the following tumor
types: astrocytoma (101), ependymoma (122), or hemangioblastoma (120). The tumor
type can be informative for the model since each type has particular characteristics, e.g.,
size, location, contrast intensity patterns, tissue constitution, (Kim et al., 2014; Baleriaux,
1999). Two modalities, Gadolinium-enhanced T1-weighted (Gd-e T1w) and T2-weighted
(T2w), are required to properly segment each component of the tumor: tumor core, edema,
and liquid-ﬁlled cavity. Here, for simplicity, only the tumor core labels were used.

2.2.2. Training scheme

The ﬁrst scenario used the FiLM architecture without any input metadata, while the second
scenario included the tumor type as metadata. To simulate the absence of metadata, the
same input vector was passed through FiLM, hence no informative data is seen by the model.
The same architecture was used in both scenarios in order to isolate the speciﬁc eﬀect of the
input metadata. Preliminary experiments gave similar results when using a regular U-Net
architecture without the FiLM layers or a FiLMed U-Net with always the same input. A
320x256 sagittal image of resolution 1mmx1mm associated with the tumor type constituted

419

Benefits of Linear Conditioning with Metadata for Image Segmentation

one training sample. The dataset was split per patient with the following proportions: 60%
training, 20% validation, 20% testing. To compare the overall segmentation performance,
10 models were trained with diﬀerent random splits.

2.3. Experiment 2: FiLM for multiple tasks

Here, the ability of FiLM to modulate the network to adapt to diﬀerent segmentation tasks
was assessed. The FiLMed model was presented with labels from three classes that are all
included in the scan, but only one segmentation was given at the time. The class of the
presented segmentation was input into the network to teach the model to properly segment
each class. A similar experiment was performed with few segmentations and unbalanced
datasets.

2.3.1. Dataset: Spleen, kidneys, and liver

The organs selected for this task were the spleen, kidneys, and liver. The datasets were
collected from two diﬀerent sources: Medical Segmentation Decathlon (Simpson et al.,
2019) for spleen and liver scans, and KiTS19 (Heller et al., 2019) for kidney scans. Liver
and kidney scans had tumor labeling which was ignored for the current experiments: organ
and tumor annotations were merged as a single segmentation. Due to the large size of the
kidney and liver datasets, subdatasets were extracted. Since the spleen dataset contained 41
scans with associated ground truths, only the ﬁrst 41 kidney and liver scans were retained.

2.3.2. Training scheme

First, the FiLMed U-Net was trained on the spleen, kidney, and liver images with the whole
dataset (41 images for each). A training example was a 2D axial slice of 512x512 pixels
paired with the available label (kidney, spleen, or liver). The dataset was split per patient
with the following proportions: 60% training, 20% validation, 20% testing.

Second, the performance on small and unbalanced datasets was assessed with an in-
dependent sub-experiment: FiLMed U-Net was trained on subdatasets of the spleen and
kidney datasets. For simplicity, only two classes were used. The experimental design of
this sub-experiment is presented in appendix A. The subdatatsets were randomly chosen
with a size of 2, 4, 6, 8, and 12 for one class and 12 subjects of the other class (i.e., a total
of 10 models: {2, 4, 6, 8, 12} spleens with 12 kidneys each and {2, 4, 6, 8, 12} kidneys
with 12 spleens each). The size of the dataset included all the subjects for training and
validation. The models were tested on 25 subjects of the class with the least subject. For a
model trained on 2 kidney subjects and 12 spleen subjects, the model would be tested on 25
kidney subjects not included in the training or validation set. During the training process,
the data was sampled to expose each class evenly to the model even when the number of
subjects is unbalanced. All the trainings were repeated 10 times with varying random splits
(100 trainings).

Regular 2D U-Nets trained on only one class at the time, spleen, kidney, or liver were

trained following the same training, validation, and test splits for comparison.

2.4. Training parameters

The tumor types or organ labels were evenly separated into three groups, training, valida-
tion, and testing groups, and the data were sampled with a batch size of 8. The FiLMed

420

Benefits of Linear Conditioning with Metadata for Image Segmentation

U-Nets of depth 4 for the spinal cord tumor and 5 for the chest CT were trained with a
Dice loss function until the validation loss plateaued for 50 epochs (early stopping with
(cid:15) = 0.001). The depth was chosen according to the size of the input images. The initial
learning rate was 0.001 and was modulated according to a cosine annealing learning rate.

2.5. Evaluation

The Dice score was selected to compare the performance of each approach. All FiLMed
approaches were compared with the conventional approach: training without informative
metadata for spinal cord tumors and on a regular U-Net for the multi-organ segmentation
tasks. To assess the statistical diﬀerences between groups, a one-sided Wilcoxon signed-rank
test with a p-value < 5% was considered to be a signiﬁcant diﬀerence.

3. Results

3.1. Experiment 1: Segmentation using relevant metadata

Prior knowledge of the tumor type led to a signiﬁcant Dice score improvement between the
regular U-Net and the FiLMed U-Net: 10.5% for the hemangioblastomas (p-value=0.006),
4.5% for the astrocytomas (p-value=0.003), and 5.1% for all tumors combined (p-value=0.003)
(Table 1). Astrocytomas and hemangioblastomas showed the highest Dice score gain when
the model was informed with the tumor type. Astrocytomas are typically large, have ill-
deﬁned boundaries, and present heterogeneous, moderate, or partial enhancement in the
Gd-e T1w contrast (Baleriaux, 1999). Conversely, hemangioblastomas are usually associ-
ated with a small tumor core (Baleriaux, 1999) intensely enhanced on Gd-e T1w (Baker
et al., 2000). These distinctive characteristics can be learned by the model to perform a
more informed segmentation (see appendix B to visualize segmentation diﬀerences).

Table 1: Spinal cord tumor core segmentation performance for regular and FiLMed U-Net (mean ±
STD % for 10 random splits). The FiLMed U-Net was trained with the tumor type as input. **
p-value < 0.05 for one-sided Wilcoxon signed-rank test.

Dice score [%]

Tumor type

No prior info.

Prior info.

Astrocytoma
Ependymoma
Hemangioblastoma

53.3 ± 4.8
57.2 ± 3.2
51.2 ± 4.0

57.8 ± 4.9 **
57.7 ± 2.4 **
61.7 ± 3.7 **

All

54.0 ± 2.2

59.1 ± 2.3 **

3.2. Experiment 2: FiLM for multiple tasks

Table 2 shows that the FiLMed multi-class model trained with missing labels (i.e., only
one organ labeled per scan) was able to reach equivalent performance to single-class U-Nets
(i.e., one model per class) trained without missing annotations. As a reference, a multi-class
2D U-net without FiLM was trained with the same dataset containing missing labels. Poor
performance was reached with an average Dice score of 41.7 ± 16.0 for all classes combined:

421

Benefits of Linear Conditioning with Metadata for Image Segmentation

Table 2: Multiple-organ segmentation Dice score with multi-class, single-class and FiLMed U-Nets
(mean ± STD %). The FiLMed U-Net was trained on spleen, kidney, and liver while regular U-Nets
were trained on each class independently. A one-sided Wilcoxon signed-rank test was performed on
columns 2 (2D U-Net) and 3 (FiLMed U-Net): no statistical diﬀerence was observed.

Our experiments

Literature

Task

Multi-class
2D U-Net

Single-class
2D U-Net

Multi-class
FiLMed U-Net

2D U-Net
(On whole challenge dataset)

Liver

50.3 ± 18.3

95.1 ± 1.4

Spleen

35.6 ± 14.2

91.7 ± 6.3

Kidney

39.2 ± 13.1

90.4 ± 9.3

94.1 ± 1.6

92.2 ± 5.3

90.7 ± 8.1

94.37 ± N/A (Isensee et al., 2018)

94.2 ± N/A (Isensee et al., 2019)

93.0 ± 1.2 (Ahmed, 2020)

only partial segmentation of each organ was performed by the model. This result illustrates
the hindered learning caused by the missing annotations. Inputting the class label through
FiLM layers allowed the model to properly train with missing segmentations enabling the
option to have a single model adapted to multiple tasks even when all annotations are not
available. For comparison, the Dice scores reached by other studies on the whole challenge
datasets, 61 spleens, 300 kidneys, and 201 livers, with 2D U-Nets was included. While being
trained on less data (41 images per dataset), our 2D FiLMed U-Net reached Dice scores
comparable with these published studies (see Table 2).

Figure 2: Spleen and kidney segmentation Dice scores for small and unbalanced datasets. The
number of subjects combines training and validation subjects. Dice scores for all experiments on the
test set (25 subjects) were averaged across the number of subjects and aggregated according to the
approach, FiLMed (red) or regular U-Net (blue). The error bars show the standard deviation. ∆
indicates the diﬀerence of mean Dice scores between the two approaches. The data totals 10 models
trained on diﬀerent random splits. ** p-value < 5% with one-sided Wilcoxon signed-rank test.

422

Benefits of Linear Conditioning with Metadata for Image Segmentation

Figure 2 demonstrates the ability of FiLMed U-Net to be trained on small or unbalanced
datasets. With the same amount of labels for a given class, FiLMed models reached superior
Dice scores for datasets of size 2, 4, 6, and 8 compared with the regular U-Nets trained on
a single class, 11.5%, 16.7%, 5.5%, and 4.7%, respectively. This suggests that the FiLMed
models were able to learn from the images associated with the other task. The more subjects
are included in the dataset, the more similar FiLM performances become to regular U-Nets,
as seen in Table 2. However, FiLMed models have the advantage of being robust to missing
classes.

4. Discussion

FiLM provides a ﬂexible, low computational cost option to integrate prior knowledge. In this
paper, the type of spinal cord tumor was exploited as a proof of concept, but the possibilities
of metadata that can improve the performance of a model are vast. The prior metadata
could include domain information (e.g., acquisition center, scanner vendor), anatomical data
(e.g., location in the body, pose estimation, disease type or severity), or rater speciﬁcation
(e.g., rater’s experience, rater’s id). To elaborate on an example, inter-expert variability
is an important aspect in medical segmentation (Renard et al., 2020).
Integrating this
information in the model would enable one to make predictions according to the rater with
the most experience or to create a model that can replicate inter-expert predictions (i.e.,
generating one prediction per expert learned in training).

FiLM is capable of dealing with missing labels by indicating which annotations are
presented to the model. Many new medical imaging datasets are available, however, most
have limited scopes and annotations. FiLM makes it possible to use data from diﬀerent
sources with only one class annotated to create a multi-class model instead of single-class
models trained on each dataset. Without the need for more labels, combining datasets
increases the number of examples seen by the model. Since weights are shared between
tasks, the model learns from the data of the other tasks as seen in Figure 2. The transfer
learning between tasks and the robustness with respect to missing segmentations reduce
the number of annotations required.

Since the metadata is one-hot encoded before being introduced into the FiLM generator,
discrete prior information is needed. The approach presented works with continuous data
(e.g., age, size, MRI acquisition parameters), but it must be discretized into a binned range.
Future work should explore methods to best encode diﬀerent data types. This enhancement
would allow the integration of MRI acquisition parameters (e.g., echo-time, ﬂip angle) that
might make the model agnostic to the diﬀerent acquisition sequences.

5. Conclusion

The integration of linear conditioning through FiLM for segmentation models enables a
ﬂexible option to integrate metadata to enhance the predictions. FiLM also facilitates the
training of multi-class models by being robust to missing labels. Future work could focus on
the impact of integrating other types of data than the tumor type, increasing the number
of metadata used to modulate the network, or evaluating the impact of including prior
information on the model’s uncertainty.

423

Benefits of Linear Conditioning with Metadata for Image Segmentation

Acknowledgments

We thank the contributors of the ivadomed project, Lucas Rouhier, Ainsleigh Hill, Valentine
Louis-Lucas, and Christian Perone for fruitful discussions.

Funded by the Canada Research Chair in Quantitative Magnetic Resonance Imaging
[950-230815], the Canadian Institute of Health Research [CIHR FDN-143263], the Canada
Foundation for Innovation [32454, 34824], the Fonds de Recherche du Qu´ebec - Sant´e
[28826], the Fonds de Recherche du Qu´ebec - Nature et Technologies [2015-PR-182754],
the Natural Sciences and Engineering Research Council of Canada [RGPIN-2019-07244],
the Canada First Research Excellence Fund (IVADO and TransMedTech), the Courtois
NeuroMod project and the Quebec BioImaging Network. This research is based on work
partially supported by the CIFAR AI and COVID-19 Catalyst Grants. A.L. has a fellow-
ship from NSERC, FRQNT, and UNIQUE, C.G. has a fellowship from IVADO [EX-2018-4],
O.V. has a fellowship from NSERC, FRQNT, and UNIQUE.

References

Mohamed Ahmed. Medical image segmentation using attention-based deep neural networks,
2020. URL https://www.diva-portal.org/smash/get/diva2:1477227/FULLTEXT01.
pdf.

Kim B Baker, Christopher J Moran, Franz J Wippold, James G Smirniotopoulos, Fabio J
Rodriguez, Steven P Meyers, and Todd L Siegal. Mr imaging of spinal hemangioblas-
toma. American Journal of Roentgenology, 174(2):377–382, 2000. URL https://www.
ajronline.org/doi/full/10.2214/ajr.174.2.1740377.

Danielle Baleriaux. Spinal cord tumors. European radiology, 9(7):1252–1258, 1999. URL

https://doi.org/10.1007/s003300050831.

Yash Bhalgat, Meet Shah, and Suyash Awate. Annotation-cost minimization for medical
image segmentation using suggestive mixed supervision fully convolutional networks. In
Neural Information Processing Systems (NeurIPS), 2018. URL https://arxiv.org/
pdf/1812.11302.pdf.

Agisilaos Chartsias, Giorgos Papanastasiou, Chengjia Wang, Colin Stirrat, Scott Semple,
David Newby, Rohan Dharmakumar, and Sotirios A. Tsaftaris. Multimodal cardiac seg-
mentation using disentangled representation learning.
In Statistical Atlases and Com-
putational Models of the Heart. Multi-Sequence CMR Segmentation, CRT-EPiggy and
LV Full Quantiﬁcation Challenges, pages 128–137, Cham, 2020. Springer International
Publishing. ISBN 978-3-030-39074-7.

Ozan Ciga and Anne L Martel. Learning to segment images with classiﬁcation labels.
Medical Image Analysis, 68:101912, 2021. URL https://doi.org/10.1016/j.media.
2020.101912.

Harm de Vries, Florian Strub, J´er´emie Mary, Hugo Larochelle, Olivier Pietquin, and
In Neural In-

Aaron C. Courville. Modulating early visual processing by language.

424

Benefits of Linear Conditioning with Metadata for Image Segmentation

formation Processing Systems (NIPS), pages 6597–6607, 2017. URL http://papers.
nips.cc/paper/7237-modulating-early-visual-processing-by-language.

Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for
artistic style. In International Conference on Learning Representations (ICLR), 2017.
URL https://arxiv.org/pdf/1610.07629.pdf.

Charley Gros, Andreanne Lemay, Olivier Vincent, Lucas Rouhier, Marie-Helene Bourget,
Anthime Bucquet, Joseph Paul Cohen, and Julien Cohen-Adad.
ivadomed: A medical
imaging deep learning toolbox. Journal of Open Source Software, 6(58):2868, 2021. URL
https://doi.org/10.21105/joss.02868.

Nicholas Heller, Niranjan Sathianathen, Arveen Kalapara, Edward Walczak, Keenan Moore,
Heather Kaluzniak, Joel Rosenberg, Paul Blake, Zachary Rengel, Makinna Oestreich,
et al. The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic
segmentations, and surgical outcomes. arXiv preprint arXiv:1904.00445, 2019. URL
https://arxiv.org/pdf/1904.00445.pdf.

Fabian Isensee, Jens Petersen, Andre Klein, David Zimmerer, Paul F Jaeger, Simon Kohl,
Jakob Wasserthal, Gregor Koehler, Tobias Norajitra, Sebastian Wirkert, et al. nnu-net:
Self-adapting framework for u-net-based medical image segmentation. arXiv preprint
arXiv:1809.10486, 2018.

Fabian Isensee, Paul F J¨ager, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein.
Automated design of deep learning methods for biomedical image segmentation. arXiv
preprint arXiv:1904.08128, 2019. URL https://arxiv.org/pdf/1904.08128.pdf.

Grzegorz Jacenk´ow, Agisilaos Chartsias, Brian Mohr, and Sotirios A Tsaftaris. Condi-
tioning convolutional segmentation architectures with non-imaging data. arXiv preprint
arXiv:1907.12330, 2019.

Grzegorz Jacenk´ow, Alison Q O’Neil, Brian Mohr, and Sotirios A Tsaftaris. Inside: Steering
spatial attention with non-imaging information in cnns. In International Conference on
Medical Image Computing and Computer-Assisted Intervention, pages 385–395. Springer,
2020.

DH Kim, J-H Kim, Seung Hong Choi, C-H Sohn, Tae Jin Yun, Chi Heon Kim, and K-H
Chang. Diﬀerentiation between intramedullary spinal ependymoma and astrocytoma:
comparative mri analysis. Clinical radiology, 69(1):29–35, 2014. URL https://doi.org/
10.1016/j.crad.2013.07.017.

Taesup Kim, Inchul Song, and Yoshua Bengio. Dynamic layer normalization for adaptive
In InterSpeech, 2017. URL https:

neural acoustic modeling in speech recognition.
//arxiv.org/pdf/1707.06065.pdf.

Andreanne Lemay, Charley Gros, Zhizheng Zhuo, Jie Zhang, Yunyun Duan, Julien Cohen-
Adad, and Yaou Liu. Multiclass spinal cord tumor segmentation on mri with deep learn-
ing, 2021. URL https://arxiv.org/pdf/2012.12820.pdf.

425

Benefits of Linear Conditioning with Metadata for Image Segmentation

Yanghao Li, Naiyan Wang, Jianping Shi, Xiaodi Hou, and Jiaying Liu. Adaptive batch
normalization for practical domain adaptation. Pattern Recognition, 80:109–117, 2018.
URL https://doi.org/10.1016/j.patcog.2018.03.005.

Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and
Demetri Terzopoulos. Image segmentation using deep learning: A survey. arXiv preprint
arXiv:2001.05566, 2020. URL https://arxiv.org/pdf/2001.05566.pdf.

Boris N Oreshkin, Pau Rodriguez, and Alexandre Lacoste. Tadam: Task dependent adaptive
metric for improved few-shot learning. arXiv preprint arXiv:1805.10123, 2018. URL
https://arxiv.org/pdf/1805.10123.pdf.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film:
Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference
on Artiﬁcial Intelligence, volume 32, 2018. URL https://arxiv.org/pdf/1709.07871.
pdf.

Michael Rebsamen, Urspeter Knecht, Mauricio Reyes, Roland Wiest, Raphael Meier, and
Richard McKinley. Divide and conquer: Stratifying training data by tumor grade im-
proves deep learning-based brain tumor segmentation. Frontiers in Neuroscience, 13:1182,
2019. ISSN 1662-453X. doi: 10.3389/fnins.2019.01182. URL https://www.frontiersin.
org/article/10.3389/fnins.2019.01182.

F´elix Renard, Soulaimane Guedria, Noel De Palma, and Nicolas Vuillerme. Variability and
reproducibility in deep learning for medical image segmentation. Scientiﬁc Reports, 10
(1):1–16, 2020. URL https://doi.org/10.1038/s41598-020-69920-0.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for
biomedical image segmentation.
In Medical Image Computing and Computer-Assisted
Intervention – MICCAI 2015, pages 234–241. Springer International Publishing, 2015.
URL https://doi.org/10.1007/978-3-319-24574-4_28.

Amber L Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani,
Bram Van Ginneken, Annette Kopp-Schneider, Bennett A Landman, Geert Litjens, Bjo-
ern Menze, et al. A large annotated medical image dataset for the development and
evaluation of segmentation algorithms. arXiv preprint arXiv:1902.09063, 2019. URL
https://arxiv.org/pdf/1902.09063.pdf.

Olivier Vincent, Charley Gros, Joseph Paul Cohen, and Julien Cohen-Adad. Automatic
segmentation of spinal multiple sclerosis lesions: How to generalize across MRI contrasts?,
2020. URL https://arxiv.org/pdf/2003.04377.pdf.

Yuyin Zhou, Zhe Li, Song Bai, Chong Wang, Xinlei Chen, Mei Han, Elliot Fishman, and
Alan L Yuille. Prior-aware neural network for partially-supervised multi-organ segmen-
tation. In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pages 10672–10681, 2019. URL https://arxiv.org/pdf/1904.06346.pdf.

426

Benefits of Linear Conditioning with Metadata for Image Segmentation

Appendix A. Experimental design of organ segmentation with limited

annotations

Figure 3: Experimental design of organ segmentation with limited annotations. The images associ-
ated to each model represent the training and validation set. This experimental design was used to
generate Figure 2.

427

Benefits of Linear Conditioning with Metadata for Image Segmentation

Appendix B. Spinal cord tumor segmentation

Figure 4: Tumor segmentation prediction by FiLMed U-Net informed by the tumor type, “With
prior”, or not informed, “No prior”. A1 and A2 presents two subjects with astrocytomas. H1 and
H2 presents two subjects with hemangioblastomas. GT: Ground truth.

Astrocytomas are typically large, have ill-deﬁned boundaries, and present heterogeneous,
moderate, or partial enhanced in the Gd-e T1w contrast (Baleriaux, 1999). Astrocytomas
are usually extensive, expanding from 2 to 19 vertebral bodies in size (Baleriaux, 1999).
In both A1 and A2 predictions from the model without prior information, the segmented
tumor size was one vertebral body or less and corresponded to the most enhanced tumor
signal on the Gd-e T1w (ignoring the rest of the lesion).

In counterpart, hemangioblastomas are usually associated with a small tumor core (Ba-
leriaux, 1999) intensely enhanced on Gd-e T1w (Baker et al., 2000). Figure 4 H1 presents
a hemangioblastoma barely apparent in T2w and hidden by the cavity (hyperintense sig-
nal). The small hyperintense signal on the Gd-e T1w contrast was overseen by the regular
approach. On H2, the model oversegmented the tumor and identiﬁed a second tumor on a
hypointense signal. The false positive tumor identiﬁcation does not present an intense Gd-e
T1w enhancement which is usually the case for hemangioblastomas. This false positive is
not present for the model informed by the tumor type.

To assess the impact of inputting the tumor type, each prediction was modulated by
the diﬀerent tumor types. Table 3 presents the quantitative results for each condition while
Figure 5 qualitatively illustrates the impact of changing the tumor type. The highest Dice
scores are reached when the input label corresponds to the true label. The modulation

428

Benefits of Linear Conditioning with Metadata for Image Segmentation

Figure 5: Impact of inputting diﬀerent tumor types with FiLMed U-Net on the model’s segmentation.
True label represents the tumor type while input label is the tumor type input into the model through
FiLM. Astr.: Astrocytoma, Epen.:Ependymoma, Hema.: Hemangioblastoma.

429

Benefits of Linear Conditioning with Metadata for Image Segmentation

Table 3: Spinal cord tumor core segmentation Dice scores for FiLMed U-Net with the diﬀerent tumor
types as input (mean ± STD % for 10 random splits). True label represents the tumor type while
input label is the tumor type input into the model through FiLM. ** p-value < 0.05 for one-sided
Wilcoxon signed-rank test compared to the highest value in each row.

Input label

True label

Astrocytoma Ependymoma Hemangioblastoma

Astrocytoma
Ependymoma
Hemangioblastoma

57.9 ± 4.9
57.6 ± 2.6
41.5 ± 4.7 **

57.3 ± 4.9
57.7 ± 2.4
41.8 ± 6.4 **

32.2 ± 5.1 **
35.9 ± 4.7 **
61.7 ± 3.7

with FiLM successfully encoded knowledge about the tumor types and the predictions are
in agreement with known characteristics of the diﬀerent types. Astrocytoma and ependy-
moma yield similar predictions. Both tumor types have overlapping characteristics (Kim
et al., 2014): high intensity signals on T2w, comparable enhancement patterns, similar
size (astrocytoma: 2-19 vertebral bodies, ependymoma: 2-13 vertebral bodies (Baleriaux,
1999)), etc. Predictions with hemangioblastoma as input diverge from the other tumor
types. Hemangioblastoma predictions reﬂect their characteristics: small tumor cores in-
tensely enhanced in Gd-e T1w, as seen in Figure 4. When inputting the hemangioblastoma
label for the astrocytoma (ﬁrst row of Figure 5) no prediction is given since the Gd-e T1w
modality has moderate enhancement. Similarly, for the ependymoma, only the most Gd-
enhanced portion of the tumor is predicted when assigning the hemangioblastoma label
with FiLM (second row of Figure 5). The results from Table 3 and Figure 4 - 5 conﬁrm
that FiLM layers are able to learn characteristics from the metadata that are relevant for
the segmentation.

430


