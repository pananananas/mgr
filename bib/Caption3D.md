Scalable 3D Captioning with Pretrained Models

Tiange Luo1,∗ Chris Rockwell1,∗ Honglak Lee1,2,† Justin Johnson1,†

1University of Michigan

2LG AI Research

Abstract

We introduce Cap3D, an automatic approach for generating descriptive text for
3D objects. This approach utilizes pretrained models from image captioning,
image-text alignment, and LLM to consolidate captions from multiple views of
a 3D asset, completely side-stepping the time-consuming and costly process of
manual annotation. We apply Cap3D to the recently introduced large-scale 3D
dataset, Objaverse, resulting in 660k 3D-text pairs. Our evaluation, conducted using
41k human annotations from the same dataset, demonstrates that Cap3D surpasses
human-authored descriptions in terms of quality, cost, and speed. Through effective
prompt engineering, Cap3D rivals human performance in generating geometric de-
scriptions on 17k collected annotations from the ABO dataset. Finally, we finetune
text-to-3D models on Cap3D and human captions, and show Cap3D outperforms;
and benchmark the SOTA including Point·E, Shap·E, and DreamFusion.

3
2
0
2

n
u
J

6
1

]

V
C
.
s
c
[

2
v
9
7
2
7
0
.
6
0
3
2
:
v
i
X
r
a

Figure 1: Cap3D provides detailed descriptions of 3D objects by leveraging pretrained models in
captioning, alignment, and LLM to consolidate multi-view information. Two views of 3D objects are
shown here, Cap3D uses eight. Additional examples are available in Appendix B.

∗ joint first authorship; † equal advising

Preprint.

3D model of a sakura soft drink can with purple and yellow gradient, Japanese writing, and purple flowers.A 3D model of a blue grand piano with spikes and sharp teeth resembling a shark mouth.A 3D model of a metal cube featuring a skull, pizza, and various stickers.3D model of a yellow Pikachu-themed Pokémon ball with a black and gold stripe and lightning bolt.3D model of Notre Dame Cathedral, a Gothic cathedral with spires in Paris.Loki bust 3D model featuring a green and yellow horned helmet.zA 3D model featuring a basketball hoop, ball, racquet, bowling ball, stand, and pin.3D model of a purple and green Halloween spider bowl on a metal stand, containing purple liquid.3D model of an armored character with purple horns and spikes on the back.3D model of a robotic scorpion with multiple arms and guns.A cluster of five glass sphere light bulbs suspended from a single thin wire.L-shaped sectional sofa with a chaise, U-shaped backrest, curved armrests, and a footstool on one side.

Table 1: Cap3D is better, cheaper, and faster than crowdsourced annotation. Use 36k responses
across 22k objects for A/B testing; 8A40s on a cloud platform for speed and cost computations.
1k Objects Cost Breakdown
Cost per
1k Objects

A/B Human Testing
Win % (Tie %)

Annotation
Speed

Method

Human
Cap3D

37.8% ± 0.5% (9.5%)
52.3% ± 0.5% (9.5%)

$87.18
$8.35

1.4k / day
65k / day

BLIP2
CLIP
GPT4

$3.79
$0.38
$4.18

Cap3D Total Cost

$8.35

1

Introduction

Text-conditioned 3D synthesis [1–3] could revolutionize the creation process of 3D assets, impacting
various sectors, including 3D design, virtual reality [4], film [5], robotics [6, 7], and autonomous
driving [8]. However, challenges persist, namely the high cost of 3D asset creation and the scarcity of
high-quality captions for 3D assets. Objaverse [9] takes a step towards this as the first public large-
scale 3D object dataset. Unfortunately, while objects contain paired metadata, these do not serve as
informative captions, as shown in Table 3. In contrast with 3D, a plethora of high-quality text-image
paired data is publicly available [10–14]. This data has led to incredible recent progress in image-text
learning [15–18], text-conditioned image synthesis [19–24], and image captioning [25–29].

In this work, we present Cap3D, a method to automate 3D object annotation. Our key insight is
to leverage the abundance of knowledge in pretrained image-text models to remedy the lack of
existing 3D-text data. The core of our data collection process is to apply an image captioning
model (BLIP2 [29]) to a set of 3D asset renders, use an image-text alignment model (CLIP [16]) to
filter captions, and apply a language model (GPT4 [30]) to fuse the filtered captions across views.
Critically, the models we apply are pretrained on varied and large-scale text-image [11–13, 31–33],
and text [34], data; and approach complementary problems. As a result, each model adds additional
value to the framework, as we show in Table 3.

Cap3D is agnostic to 3D asset sources and can be effectively scaled to larger extents with increased
3D assets and computational resources. In this paper, we apply it primarily to Objaverse, gathering a
dataset of 660k 3D-text pairs. Through object rendering and captioning, we enable ethical filtering
of 3D objects via both image and text, as detailed in § 3.2. We publicly release all of our collected
data including automated and human-annotated captions, along with associated Point Clouds and
Rendered Images, at huggingface.co/datasets/tiange/Cap3D. The dataset is released under
ODC-By 1.0 license. We will release trained models and code for replicating the benchmark table.

We validate our collection approach by collecting over 50k crowdsourced captions on over 40k
objects. We conduct human evaluations and show on Objaverse that our automated captions are
superior to crowdsourced captions in quality, cost, and speed (Table 1, details in Appendix A).
Specifically, it is preferred 35% more often by humans, costs more than 10 times less, and is over
40 times faster, assuming only 8A40 GPUs. We also test the limits of automated captioning. We
consider a separate task of captioning geometry (as shown in Figure 1 bottom-right) using ABO, a
dataset of 3D models with complex geometries [35]. Shown in Table 4, our automated captioning
underperforms humans. However, by formulating description as a question answering task (detailed
in § 3.1), we show stronger performance compared to crowdsourced workers. This result shows the
ability of our method to adapt beyond traditional captioning and still be highly competitive.

Finally, our high-quality gathered 3D-text dataset enables us to train and validate large-scale text-
to-3D models. In §5.3, we evaluate several state-of-the-art methods on Objaverse out-of-the box,
including Point·E, Shap·E, DreamFields, and DreamFusion. Finetuning on our data typically shows
meaningful improvements, demonstrating the value of the collected dataset. In addition, we show our
automatically collected captions yield better finetuning performance than human captions – even at
the same scale. At full scale, finetuning is further boosted.

2 Related Work

Obtaining 3D-text pairs at scale is challenging, and we take inspiration from image-text datasets and
methods when approaching this task.

2

Image-Text Data and Modeling. Early image captioning [36–38] and text-image representation
learning methods [39, 40] were built using CNNs [41–43] and LSTMs [44, 45], leveraging human-
annotated datasets [31–33, 46]. Text-to-image methods used similar datasets, and relied on GANs [47,
48] and VQVAEs [19, 49–51]. The advent of semi-automated image-text collection has enabled
successful scaling of datasets [10–14] and models [25–28]. Transformer-based architectures [16,
52, 53] and diffusion models [54–59] have scaled best to large data; we employ transformer-based
methods through our captioning process and adopt diffusion models for text-to-3D experiments.

Training models upon large datasets and using the corresponding trained models to filter larger data
has led to datasets of rapidly increasing size [13, 14]. In addition to filtering, trained models have
been used to annotate new data with high-quality [60]. We take this approach, captioning rendered
views with BLIP2 [29], refining with CLIP [16, 61], and summarizing with GPT4 [62]; all of which
are trained on large datasets, including [11–13, 31–33]. Concurrent works [63–65] use automated
captioning on 2D images using an older system [66] or based upon metadata [65, 67].

3D-Text Data and Modeling. Until recently, 3D data was of relatively small scale (∼ 50k objects)
[68–71]. Labeled 3D-text data was scarce, relying on human annotation, and typically limited to
ShapeNet [68] chairs [72] or tables and chairs [73, 74], and ScanNet [75, 76]. This enabled prior work
to undertake the task of 3D captioning [77–79] or text-to-3D [73, 77, 80–83] at small scale. Methods
that approached text-to-3D would sometimes avoid 3D supervision entirely [3, 84–86], leading to
slow generation due to many optimization steps. We annotate a small-scale dataset containing 3D
furniture, ABO [35], to evaluate the ability of Cap3D to specify fine-grained geometry.

Objaverse [9] introduced a diverse set of objects over 10 times the size of the prior largest public 3D
dataset [68]. This data is our primary captioning focus, and we associate a single caption with each
object in Objaverse after filtering. Concurrent works [65, 87] gather text associated with Objaverse,
but do not fuse captions across views [87] or rely upon metadata [65], and do not approach text-to-3D.

The concurrent studies 3DGen [2] learns text and image to 3D on Objaverse; Point·E [88] and
Shap·E [89] learn text-to-3D models on a large-scale 3D dataset, but none have fully disclosed their
code or data. Point·E involves two variants and released a text-to-3D model and a text-to-image-to-3D
model by finetuning GLIDE [23] and training an image-to-point cloud diffusion model [90]. Other
recent works [91, 92] also focus on scaled image-3D generation. We show finetuning on our captions
improves Point·E performance despite having already been trained on large amounts of Internet data.

3 Method

3.1 Captioning Process

Our task is to produce a single descriptive caption given a 3D asset. Our proposed method, Cap3D,
employs a four-step process. First, we render a set of 2D views for each 3D object. Next, we apply
image captioning to achieve preliminary descriptions. As these captions may contain inaccuracies, an
image-text alignment model, CLIP, is introduced in the third step to rectify errors. Finally, an LLM
is employed to unify captions from various perspectives, creating a comprehensive caption. This
process is shown in Figure 2 and detailed below.

Object Rendering: We render using Blender at 512×512 from M = 8 high-information camera
angles rotating horizontally around the object, with two slightly below and the rest slightly above
the object, to cover all the object details. The reason we prefer multiple views is a forward-facing
view may miss self-occluded object details (e.g. Figure 1 row 1) or face strange appearance and/or
lighting. In contrast, multiple views will see much of the object from different viewpoints, increasing
the number of chances for a captioning model to predict objects in detail. For instance, in Figure 2,
the back view 1 identifies the "yellow handle", which is barely visible in forward view M .

Image Captioning: We use BLIP2 [29] for captioning, selecting the largest pretrained model
adapting ViT-G [53, 93] image encoder and FlanT5XXL [94] text encoder. We generate N = 5
captions per rendered image using nucleus sampling [95]. By generating multiple captions, we
increase the likelihood of generating correct details (e.g. "black and yellow toy bomb" in Figure 2
view M caption 1). Incorrect captions, such as "scissors" in Figure 2 view M caption N , can be
filtered in later stages. To generate captions containing fine-grained geometry details (in our ABO
experiments), we employ a two-stage question-answering instead of captioning. The first stage

3

Figure 2: Overview of Cap3D. Left to Right: (1) Render 3D objects from M = 8 camera angles
to capture object details (2) Generate N = 5 image captions per rendered image using BLIP2; (3)
Select one caption for each image based on its similarity to the image encoding using CLIP; (4) Use
GPT4 to consolidate all selected captions into a final, summary of the object.

generates one answer to a prompt asking what object is pictured. The answered object is passed into
a second prompt, which asks its structure and geometry, and generates 5 answers.

Caption Selection: While BLIP2 often generates high-quality captions, it is not uncommon for
samples to contain mistakes, particularly in non-forward facing views such as "yellow cup", in Figure
2 view 1, caption N . To reduce the frequency of mistakes, we compute CLIP [16] ViT-B/32 [53]
encodings from each of 5 captions and the associated image, and select the caption maximizing cosine
similarity. CLIP tends to select good captions for each view, e.g. Figure 2: view 1, BLIP2 caption 1
and view M , caption 1. CLIP is complementary to BLIP2 as not only does it have different training
details and architecture, but it trains on different data. While BLIP2 is trained upon COCO [31],
Visual Genome [32], CC3M [11], CC12M [12], SBU [33] and LAION400M [13]; CLIP is trained
upon a dataset of 400M images based on frequent text occurrence in Wikipedia.

Caption Consolidation: Accumulating information across viewpoints to form a complete picture of
3D objects is challenging, but crucial. We find prompting of GPT4 [62] to summarize the M captions
results in good parsing of the details across captions. By applying GPT4 as the final summary step,
it can both include significant details and remove unlikely ones. For example, the final caption in
Figure 2 filters the incorrect information, from view 2, “toy ball", while keeping key details, including
"handle" and "straw". The alternative order of GPT4 followed by CLIP would result in (1) GPT4
having to make sense of more incorrect input details and (2) CLIP simply selecting between aggregate
captions instead of being able to error-correct small mistakes. The effectiveness of introducing GPT4
is verified in ablations (Table 3).

3.2 Ethical Filtering

Captions generated and images rendered by Cap3D enhance the identification and mitigation of legal
and ethical issues associated with large-scale 3D object datasets, including identifiable information
and NSFW content.

We manage two datasets: Objaverse and ABO. In Objaverse, our main responsibility involves dealing
with artist-created assets. These can include identifiable elements such as human face scans and
NSFW objects. Objaverse contains approximately 800k objects, which makes the manual verification
of each asset impractical. The ABO dataset, on the other hand, is smaller and mostly consists of
furniture. We manually ensure the ethical integrity of this dataset.

We begin by filtering Objaverse to include only those objects that can be rendered and shared. Objects
with CC BY-NC-SA and CC BY-NC licenses are removed, while we retain those with CC BY, CC
BY-SA, and CC0 licenses, thereby facilitating commercial usage of our data. This process reduces
the dataset size from 798k to 723.7k objects. Furthermore, we exclude objects that lack sufficient
camera information for rendering, leaving us with 680k objects.

We next follow prior work [10] and use a face detector [96] and NSFW classifier [97, 98] on forward-
facing object renders and filter detected objects with score >= 0.9. The face detector filters out 18.6k

4

GPT4CLIPA 3D model of a yellow and black toy bomb with a handle and a strawInput 3D Object…1. A black and yellow toy bomb on a grey background N. A 3d model of a bomb with scissors…View 1View MOutput CaptionView 1. A yellow and black bomb with a yellow handleView 2. a yellow and black toy ball with a straw sticking out of it View M. A black and yellow toy bomb on a grey background…1. A yellow and black bomb with a yellow handleN. 3d model of a bomb with a yellow cup on it……BLIP2…BLIP2CLIPPrompt:Given a set of descriptions about the same 3D object … distill these descriptions into one concise caption:Figure 3: Objaverse Caption Comparison. Human captions and Internet metadata frequently
contain limited detail. Cap3D captions typically have longer length and more detail.

objects, and the NSFW classifier filters out 217 objects. Text is also carefully processed. Our final cap-
tions are the output of GPT4, which has been trained to filter out inappropriate or harmful content [62].
We run a standard blocklist [99] on its output, remov-
Table 2: Ethical Filtering Analysis. We
ing any object-caption pairs including blocked words.
manually detect faces and NSFW content
This filters out 226 objects. After all the filtering, we
to validate automated filtering. 16 of 17
are left with 661k objects in the Objaverse dataset. We
missed face detections were sports cards.
manually estimate detection precision and recall in Ta-
Detected Precision Missed dets.
ble 2. To summarize, our process detects over 19k
objects, of which a nontrivial amount is accurately re-
moved. We estimate roughly 1k face and less than 1k
NSFW are missed, using a conservative standard (e.g.
missed faces are typically sports cards).

18.6k
Faces
NSFW
217
Language † 226

790 16% 17 ≈1k
102 47% 12 <1k
–
–
†: String match filtering is deterministic.

10k 680k

(Filtered)

5k (%)

–

–

4 Dataset

We collect captions in two distinct settings: Objaverse, a large and varied dataset of artist-created 3D
assets; and ABO, a small dataset of real products, typically furniture.

4.1 Objaverse Captions

Objaverse [9] features roughly 800k 3D object assets across 21k classes designed by over 100k artists.
It is of significantly larger scale than prior work; the paper shows this size enables more diversity
by generative 3D models trained upon it. It is released under the ODC-By 1.0 license, permitting
subsequent researchers to curate new data from it. Metadata is paired with many assets, however as
seen in Figure 3 (right), metadata caption length is frequently short or empty. We collect two caption
datasets on Objaverse. First, an automated set of one caption for each of 660k objects using Cap3D
(a total of 660k captions). Second, a crowdsourced set of 41.4k captions spanning 39.7k objects
for evaluating generated captions. Captions are collected using thehive.ai, a crowdsourced platform
similar to AMT. Workers are given instructions with gold-standard sample captions, see the same
8 views as models during captioning, and are routinely monitored. Poor captioning performance
results in a ban and deletion of the worker’s captions. Crowdsourced captions are also filtered using
the blocklist in § 3.2. Figure 3 (left) shows human captions provide more detail than metadata, but
automated captions tend to be most descriptive.

4.2 ABO Geometry Captions

ABO [35] is a collection of 3D models of Amazon products and is primarily furniture. ABO serves
as an important contrast to Objaverse as it consists of a small number of classes varying primarily in
geometry. Captioning, therefore, needs to focus more on structure as opposed to semantic category.
To emphasize this focus, we consider the task of captioning the geometric structure of objects without
color or texture (seen in the bottom right of Figure 1). Like Objaverse, ABO contains metadata that is
typically quite short (Table 4), resulting in limited detail. We collect three sets of captions on the
6.4k ABO splits of [77]: crowdsourced (a total of 17.2k captions), captions generated by Cap3D

5

A plant with white flowersFlower.MetadataHumanCap3D3D model of a branch with white flowers and green leaves.051015202530Number of words in caption06121824% of captionsMetadataHumanCap3DFigure 4: ABO Automated Geometric Description. Left: Human descriptions provide more
detailed geometry than automated captions. With careful prompting, Cap3D (QA) can match human-
level detail. Right: The high peak of Metadata is cropped, which otherwise obscures other curves.

(a total of 6.4k captions), and captions generated by Cap3D (QA) which uses the two-stage prompt
captioning (a total of 6.4k captions). Crowdsourced captions follow similar detail to Objaverse with
the exception instructions and examples are focused on geometric structure. We compare alternatives
in Figure 4. In contrast to Objaverse, human geometric descriptions on ABO are more detailed than
captioning. With prompting (QA), the Cap3D pipeline can rival human descriptions.

5 Experiments

In this section, we first validate the quality of Cap3D captions against metadata and human-authored
captions on both Objaverse and ABO. To verify Cap3D captions are helpful in practice, we next
compare text-to-3D models finetuned on both human-authored captions and Cap3D (using the same
>30k set as crowdsourced captions). Finally, we evaluate state-of-the-art text-to-3D models on our
captions at scale to measure if finetuning on our captions can improve performance.

5.1

3D Captioning on Objaverse

Dataset. We evaluate caption quality on three subsets of Objaverse: (1) a random set of 22k objects
containing a human caption, (2) a random split of 5k objects containing a human caption, and (3) a
random 5k split across the entire dataset.

Baselines. In data splits (1) and (2), we compare the caption generated by Cap3D with human-
authored annotations, Human, and existing Objaverse metadata, Metadata, described in § 4.1. Split
(1) is used for A/B testing of Cap3D vs. Human, as shown in Table 1, at scale. Collecting A/B
comparison is expensive, so we compute more extensive experiments on the smaller set (2) in Table 3.

In data split (3), we ablate the main components of Cap3D into BLIP2 and +GPT4. BLIP2 uses only
the image captioning component of our method, taking a front-view rendering and producing a single
output caption. +GPT4 uses the same image captioning process of our method, producing 5 captions
for each of 8 views. However, instead of using CLIP to filter 5 captions from each view, it directly
summarizes all 40 captions into a final caption.

Metrics. Our primary metric is human judgment A/B tests, where we ask workers to select between
two captions on a scale of 1-5, where 3 is a tie. Workers are carefully monitored and each comparison
has at least 10k observations across 5k objects.We report mean score, along with the percent each
method is preferred (i.e. scores a 4 or 5). We use automated metrics CLIPScore [16, 61], the cosine
similarity of CLIP encodings with input images; and ViLT Image and Text Retrieval, which ranks
likely image-text pairs, from which one computes precision.

We emphasize CLIPScore is not our primary metric since our captioning model utilizes CLIP. BLIP2
utilizes ViT-L/14 and ViT-g/14, while our filtering uses ViT-B/32, so following previous work [84]
we compute CLIP score using a different model to reduce bias (ViT-B/16). However, we report it as it
has shown a higher correlation with human judgments than other automated metrics [61]. ViLT [100]
is trained on different data and is a different architecture than CLIP, providing an orthogonal metric.

6

Bed.Cap3DMetadataHuman3D rendering of a couch.Cap3D (QA)Three-seater sofa with a slender, curved backrest and armrests.A three seater sofa with low backrest to the height of the armrests.051015202530Number of words in caption06121824% of captionsMetadataHumanCap3DCap3D (QA)Table 3: Objaverse Captions Evaluations. Cap3D outperforms human and Metadata; BLIP2, GPT4,
and CLIP are all important to performance. We report 95% confidence interval and use 5k objects.
CLIP ViLT Img Retr. ViLT Text Retr.
R@10
Score R@5 R@10 R@5

User A/B Study vs. Cap3D
Win %

Score (1-5)

Method

Lose %

Metadata
Human
Cap3D

1.74±0.026
2.86±0.026
-

10.7 ± 0.7
37.0±1.0
-

83.8 ± 0.8
46.1±1.0
-

BLIP2
+ GPT4
+ CLIP (Cap3D)

2.87± 0.019
2.94± 0.015
-

41.0± 0.7
35.2± 0.6
-

50.6± 0.7
40.8± 0.6
-

66.8
72.5
88.4

83.1
86.3
86.9

4.3
21.2
35.7

24.7
31.9
31.1

6.3
29.0
46.3

32.3
39.9
40.2

6.1
18.5
34.7

21.9
30.2
30.3

8.5
24.9
44.2

29.3
38.4
38.6

Figure 5: Objaverse Caption Ablations. GPT produces longer and more detailed captions than
BLIP2; CLIP tends to prune incorrect details and reduces length slightly.

Results. We report large scale A/B testing (1) against Human in Table 1, which shows Cap3D is
better across metrics, with high confidence. The top three rows of Table 3 use the smaller human-
captioned split (2), and demonstrate Cap3D’s superior performance over Objaverse metadata and
human-authored captions across A/B studies and automated metrics. The bottom three rows of
Table 3, studied across a random split of the full dataset (3), reveal that while BLIP2 is effective,
incorporating multiple views with +GPT4 enhances performance. As shown in Figure 5, GPT4 adds
detail by consolidating view-specific information. Filtering using +CLIP (Cap3D) mitigates false
details by purging subpar captions from GPT input. In addition to reducing errors, utilizing CLIP
also reduces GPT input captions from 40 to 8, effectively decreasing token numbers and facilitating a
cost reduction from $15.33 to $4.18.

5.2 Geometry 3D Captioning on ABO

Dataset. We evaluate geometric captioning on a 6.4k object split from ABO [35, 77], comparing
Cap3D captions for each object against a maximum of two human-authored ones. To emphasize
geometric focus, images used for model input and human assessment are texture-free and colorless.

Baselines and Metrics. We use two automated variants from §3.1: Cap3D and Cap3D (QA), which
uses a two-stage prompt captioning to ask more about the input 3D geometry; and compare to
crowdsourced human descriptions, Human, detailed in §4.1, and ABO metadata, Meta.

Our primary metric of comparison is similar human A/B testing to §5.1, since automated metrics
such as CLIPScore do not accurately represent the distance between fine-grained captions and images
as shown in [77].

In stark contrast to Objaverse, Human captions beat automated (Cap3D) in Table 4.
Results.
Automated captions alone contain little geometric detail (e.g., Figure 4), making Cap3D unsuited
for this setting. However, by using the two-stage prompt engineering, Cap3D (QA) is preferred to
Human. Shown in Figure 4, Cap3D (QA) produces significant fine-grained geometric detail as well
as longer captions in general. In contrast, Metadata is clearly the weakest baseline.

7

A 3D model of a house with a garage, roof, grass and trees.A 3D model of a house with a roof, garage, grass, trees, and a green field, featuring a knife and a pair of scissors.A 3D model of a house on a gray background.+ GPT4BLIP2+ CLIP (Cap3D)051015202530Number of words in caption06121824% of captionsBLIP2+ GPT4+ CLIP (Cap3D)Table 4: ABO Fine-Grained Geometry Cap-
tions. Cap3D (QA) performs best; crowd-
sourced beats captioning alone.

Method

A/B

A/B

Score (1-5) Win %

A/B
Lose %

3.09±0.02
Human v. Cap3D
3.08±0.02
Cap3D(QA) v. Human
Cap3D(QA) v. Cap3D 3.27±0.02
4.27±0.02
Cap3D(QA) v. Meta

47.3±1% 41.4±1%
50.2±1% 44.0±1%
56.0±1% 37.4±1%
88.2±1% 10.0±1%

Table 5: Text-to-3D: Human Captions. Cap3D
captions are better than human on the 30k set.
Finetuning on Cap3D full set performs best.

Finetune
Dataset

FID↓

CLIP CLIP R-Precision (2k)
Score R@1 R@5 R@10

E

·
t
n
i
o
P

·

E
p
a
h
S

Pretrained
30k (Human)
30k (Cap3D)
350k (Cap3D)

Pretrained
30k (Human)
30k (Cap3D)
350k (Cap3D)

36.1
34.6
33.7
32.8

37.2
36.0
37.2
35.5

72.4
74.4
75.0
75.6

80.4
79.6
79.4
79.1

6.0
8.2
10.4
12.4

20.3
18.6
19.1
20.0

16.2
21.3
24.3
28.1

39.7
36.3
37.5
38.8

22.4
29.1
32.1
36.9

48.7
45.3
46.1
47.3

5.3 Large-Scale Text-to-3D Generation

Dataset. We evaluate text-to-3D generation on three subsets of Objaverse: (1) a 30k split of objects
containing human-authored captions, to measure if finetuning on Cap3D captions outperform human-
authored ones; (2) a 350k split of Objaverse objects paired with Cap3D captions, for finetuning
state-of-the-art text-to-3D methods – obtaining high-density point cloud and latent codes to finetune
Point·E and Shap·E for all 660k objects is prohibitively expensive (20k GPU days); and (3) a 300
object split for optimization-based baselines, which typically take >30 mins per object to optimize.
Pretrained and Finetuned models are evaluated on 8 views across a held-out test set of 2k objects.

Methods. We consider several recent SOTA methods in three general categories: text-to-3D diffusion,
cascaded text-to-image then image-to-3D diffusion, and optimization-based. We use the direct
text-to-3D variant of Point·E [88], as well as two variants of Shap·E [89]: STF [101] and NeRF [102].
We use Stable Diffusion cascaded with Point·E (Im-to-3D), adapting ControlNet [63] and LoRA [103]
for Stable Diffusion finetuning. We use optimization-based baselines DreamField [84], the publicly
available implementation of DreamFusion [3], Stable DreamFusion [104]; and 3DFuse [105], using
their implementation based on Karlo [24, 106].

Metrics. We use standard metrics from prior work [3, 84, 88, 89] to evaluate. Primarily, these are
CLIP Score and CLIP R-Precision. CLIP R-Precision ranks a rendered image against all text pairs in
the test set by CLIP cosine similarity, and computes precision upon true text-image correspondence.
Since we have ground truth images, we calculate the FID [107] of 3D rendered images against ground
truth images, as well as assess CLIP Score on these reference images. We also use ViLT Retrieval
R-Precision, used in 5.1, which has the same evaluation procedure as CLIP R-Precision with a
different model.

Results. Table 5 lists the results of finetuning using human-authored and Cap3D captions. Point·E
improves after finetuning upon human captions. However, performance is further improved using our
captions on the same dataset; and improved most by training upon the full dataset. This result strongly
defends Cap3D captioning at scale. Shap·E does not improve on CLIP metrics after finetuning in any
dataset, but performs the least bad on the full dataset using our captions; and FID improves most.

Table 6 presents results from several state-of-the-art pretrained and finetuned models using Cap3D-
generated captions. The models finetuned on our captions generally outperform pretrained models
under the FID metric. For CLIP-related metrics, the finetuned models of Point·E (Text-to-3D) and
StableDiffusion + Point·E (Im-to-3D) also beat their pretrained counterparts. Point·E and Stable
Diffusion have been trained on massive datasets, so improvement from finetuning is strong evidence
Cap3D captions are effective. The observed downturns in Shap·E could be attributed to at least two
factors. First, our replication of their privately-available train code is unstable, often resulting in NaN
loss during finetuning. We restart from earlier checkpoints upon crashing, but the result alone is
concerning. Second, we exclusively finetune the diffusion model in Shap·E’s two-stage approach.

Qualitative results in Figure 6 validate quantitative findings. Point·E and Stable Diffusion baselines
show large improvements from finetuning, while Shap·E can better fit the Objaverse data distribution
(corresponding to improved FID).

8

Table 6: Text-to-3D on Objaverse. Finetuning improves FID over pretrained performance across
models. CLIP metrics of Stable Diffusion increase; CLIP metrics of Point·E increase significantly.

Pretrained

FID↓

CLIP CLIP R-Precision (2k)
Score R@1 R@5 R@10

FID↓

Finetuned on Cap3D
CLIP CLIP R-Precision (2k)
Score R@1 R@5 R@10

Ground Truth Images

Point·E (Text-to-3D) [88]
S. Diff. [22] (CNet) [63]+ [88](Im-to-3D)
S. Diff. [22] (LoRA) [103]+ [88](Im-to-3D)
Shap·E [89] (STF) [101]
Shap·E [89] (NeRF) [102]

-

36.1
54.7
54.7
37.2
48.7

81.6

72.4
73.6
73.6
80.4
79.4

32.7

6.0
11.0
11.0
20.3
19.0

55.1

16.2
23.4
23.4
39.7
37.7

64.3

22.4
30.0
30.0
48.7
46.8

-

32.8
53.3
53.7
35.5
48.2

81.6

75.6
74.6
74.4
79.1
78.1

32.7

12.4
12.4
11.6
20.0
18.3

55.1

28.1
26.2
24.6
38.8
35.1

64.3

36.9
33.8
31.4
47.3
43.5

Figure 6: Text-to-3D results. Finetuning on Cap3D captions can significantly improve results.

Optimization baselines, shown in Table 7, per-
form very well upon CLIP-based metrics, con-
sistent with prior work [89]. In fact, DreamField
outperforms ground truth images in CLIP met-
rics. This demonstrates DreamField overfits to
the CLIP metric, which is the standard protocol
for text-to-3D evaluation. We propose to also
consider ViLT precision (see §5.1). This helps
mitigate the bias of CLIP, though DreamField
performance on this metric is still strong.

Table 7: Text-to-3D: Optimization Baselines.
Overfitting via CLIP leads to higher CLIP-based
scores than ground truth; ViLT score is more fair.

FID↓

CLIP

ViLT

Score R@1 R@5 R@1 R@5

True Images

-

D. Field [84]
D. Fusion [3]
3DFuse [105]

106.1
127.8
93.4

83.2

83.7
72.4
75.8

53.2

61.8
28.4
38.8

77.8

83.6
46.1
59.5

41.3

32.3
23.7
24.7

69.0

56.0
45.3
51.0

6 Conclusion

In this work, we collect (1) 3D object captions at scale, creating the largest publicly available high-
quality 3D-text by an order of magnitude. To do so we propose Cap3D, an automated pipeline
leveraging several models pretrained on large datasets, and show design choices are important to
performance. In addition, we collect (2) a dataset of geometric captions upon fine-grained 3D
objects. This helps analyze shortcomings of automated captioning and study the potential of question
answering, while yielding geometric descriptions for 3D assets of real objects paired with real images.
These datasets serve as benchmarks for text-to-3D tasks (1) at scale and (2) in geometric detail.

Acknowledgments and Disclosure of Funding

This work is supported by two grants from LG AI Research and Grant #1453651 from NSF. We
greatly thank Kaiyi Li for his technical support. We thank Mohamed EI Banani, Karan Desai, and Ang
Cao for their helpful discussions. Thanks Matt Deitke for helping with Objaverse-related questions.

9

A 3D white skateboard ramp model.A 3D model of a green teapot with horns.FinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)References

[1] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint

arXiv:2305.02463, 2023.

[2] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion

for textured mesh generation. arXiv, 2023.

[3] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d

diffusion. arXiv, 2022.

[4] Yuk Ming Tang and Ho Lun Ho. 3d modeling and computer graphics in virtual reality. In Mixed Reality

and Three-Dimensional Computer Graphics. IntechOpen, 2020.

[5] Rick Parent. Computer animation: algorithms and techniques. Newnes, 2012.

[6] Afsoon Afzal, Deborah S Katz, Claire Le Goues, and Christopher S Timperley. A study on the challenges

of using robotics simulators for testing. arXiv preprint arXiv:2004.07368, 2020.

[7] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín,
Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for
embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning,
pages 80–93. PMLR, 2023.

[8] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open

urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

[9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig
Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d
objects. 2023.

[10] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data

created by the people, for the people. NeurIPS, 2021.

[11] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned,

hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.

[12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-
scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021.

[13] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush
Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered
400 million image-text pairs. arXiv, 2021.

[14] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti,
Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale
dataset for training next generation image-text models. 2022.

[15] Mohamed El Banani, Karan Desai, and Justin Johnson. Learning Visual Representations via Language-

Guided Sampling. In CVPR, 2023.

[16] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from
natural language supervision. In ICML, 2021.

[17] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets language-

image pre-training. In ECCV, 2022.

[18] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc,
Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for
few-shot learning. NeurIPS, 2022.

[19] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and

Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.

[20] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene:

Scene-based text-to-image generation with human priors. In ECCV, 2022.

10

[21] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-
image diffusion models with deep language understanding. 2022.

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution

image synthesis with latent diffusion models. In CVPR, 2022.

[23] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya
Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided
diffusion models. CoRR, 2021.

[24] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional

image generation with clip latents. arXiv, 2022.

[25] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple

visual language model pretraining with weak supervision. ICLR, 2022.

[26] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong
Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pre-training for vision-language tasks. In
ECCV, 2020.

[27] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and
Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In CVPR, 2021.

[28] Zhizhong Han, Chao Chen, Yu-Shen Liu, and Matthias Zwicker. Shapecaptioner: Generative caption
network for 3d shapes by learning a mapping from parts detected in multiple views to sentences. In ACM
MM, 2020.

[29] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-

training with frozen image encoders and large language models. arXiv, 2023.

[30] OpenAI. Gpt-4 technical report, 2023.

[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,

and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.

[32] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen,
Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision
using crowdsourced dense image annotations. IJCV, 2017.

[33] Vicente Ordonez, Girish Kulkarni, and Tamara Berg.

Im2text: Describing images using 1 million

captioned photographs. NeurIPS, 2011.

[34] https://commoncrawl.org/the-data/.

[35] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang,
Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik.
Abo: Dataset and benchmarks for real-world 3d object understanding. CVPR, 2022.

[36] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei
Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In CVPR,
2018.

[37] Steven J Rennie, Etienne Marcheret, Youssef Mroueh, Jerret Ross, and Vaibhava Goel. Self-critical

sequence training for image captioning. In CVPR, 2017.

[38] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Neural baby talk. In CVPR, 2018.

[39] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet:

Modular attention network for referring expression comprehension. In CVPR, 2018.

[40] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for

image-text matching. In ECCV, 2018.

[41] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.

In CVPR, 2016.

[42] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional

neural networks. In Advances in Neural Information Processing Systems, 2012.

11

[43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical

image segmentation. In MICCAI 2015, 2015.

[44] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 1997.

[45] Mike Schuster and Kuldip K Paliwal. Bidirectional recurrent neural networks. transactions on Signal

Processing, 1997.

[46] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi
Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In ICCV, 2019.

[47] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron

Courville, and Yoshua Bengio. Generative adversarial nets. 2014.

[48] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing

and improving the image quality of stylegan. In CVPR, 2020.

[49] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 2017.

[50] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image

synthesis. In CVPR, 2021.

[51] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou
Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. NeurIPS,
2021.

[52] Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In

CVPR, 2021.

[53] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is
worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

[54] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021.

[55] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33, 2020.

[56] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based

generative models. NeurIPS, 2022.

[57] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In

ICML, 2021.

[58] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.

NeurIPS, 2019.

[59] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised

learning using nonequilibrium thermodynamics. In ICML, 2015.

[60] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao,

Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv, 2023.

[61] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free

evaluation metric for image captioning. arXiv, 2021.

[62] OpenAI. Gpt-4 technical report. arXiv, 2023.

[63] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models.

arXiv, 2023.

[64] Justin N. M. Pinkney. Pokemon blip captions. https://huggingface.co/datasets/lambdalabs/

pokemon-blip-captions/, 2022.

[65] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih
Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding.
arXiv, 2023.

[66] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training

for unified vision-language understanding and generation. In ICML, 2022.

12

[67] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Car-
los Niebles, and Silvio Savarese. Ulip: Learning unified representation of language, image and point
cloud for 3d understanding. CVPR, 2023.

[68] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio
Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository.
arXiv, 2015.

[69] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B
Tenenbaum, and William T Freeman. Pix3d: Dataset and methods for single-image 3d shape modeling.
In CVPR, 2018.

[70] Joseph J Lim, Hamed Pirsiavash, and Antonio Torralba. Parsing ikea objects: Fine pose estimation. In

ICCV, 2013.

[71] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao.

3d-future: 3d furniture shape with texture. IJCV, 2021.

[72] Panos Achlioptas, Judy Fan, X.D. Robert Hawkins, D. Noah Goodman, and J. Leonidas Guibas. Shape-

Glot: Learning language for shape differentiation. CoRR, 2019.

[73] Kevin Chen, Christopher B Choy, Manolis Savva, Angel X Chang, Thomas Funkhouser, and Silvio
Savarese. Text2shape: Generating shapes from natural language by learning joint embeddings. In ACCV,
2019.

[74] Rao Fu, Xiao Zhan, Yiwen Chen, Daniel Ritchie, and Srinath Sridhar. Shapecrafter: A recursive
text-conditioned 3d shape generation model. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and
Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https:
//openreview.net/forum?id=KUOKpojFr_.

[75] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner.

Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

[76] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d

scans using natural language. In ECCV, 2020.

[77] Tiange Luo, Honglak Lee, and Justin Johnson. Neural shape compiler: A unified framework for
transforming between text, point cloud, and program. Transactions on Machine Learning Research, 2023.
ISSN 2835-8856. URL https://openreview.net/forum?id=gR9UVgH8PZ.

[78] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image

caption generator. In ICML, 2015.

[79] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X. Chang. Scan2cap: Context-aware dense

captioning in rgb-d scans. In CVPR, 2021.

[80] Aditya Sanghi, Hang Chu, Joseph G Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, and Ka-
mal Rahimi Malekshan. Clip-forge: Towards zero-shot text-to-shape generation. In CVPR, 2022.

[81] Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, and Shubham Tulsiani. Autosdf: Shape priors for 3d

completion, reconstruction and generation. In CVPR, 2022.

[82] Jiacheng Wei, Hao Wang, Jiashi Feng, Guosheng Lin, and Kim-Hui Yap. Taps3d: Text-guided 3d textured
shape generation from pseudo supervision. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 16805–16815, 2023.

[83] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representa-

tion for neural fields and generative diffusion models. arXiv preprint arXiv:2301.11445, 2023.

[84] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided

object generation with dream fields. In CVPR, 2022.

[85] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari.

Textmesh: Generation of realistic 3d meshes from text prompts. arXiv, 2023.

[86] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis,
Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309,
2023.

13

[87] Le Xue, Ning Yu, Shu Zhang, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu,
Juan Carlos Niebles, and Silvio Savarese. Ulip-2: Towards scalable multimodal pre-training for 3d
understanding. arXiv, 2023.

[88] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for

generating 3d point clouds from complex prompts. arXiv, 2022.

[89] Alex Nichol and Heewoo Jun. Shap-e: Generating conditional 3d implicit functions. arXiv, 2023.

[90] Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel diffusion.
In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5826–5835, 2021.

[91] Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari. Multiview

compressive coding for 3d reconstruction. arXiv, 2023.

[92] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick.

Zero-1-to-3: Zero-shot one image to 3d object. arXiv, 2023.

[93] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong
Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. CVPR,
2023.

[94] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang,
Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv,
2022.

[95] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text

degeneration. In ICLR, 2020.

[96] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-

shot multi-level face localisation in the wild. In CVPR, 2020.

[97] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the

inception architecture for computer vision. In CVPR, 2016.

[98] Gant Laborde. Deep nn for nsfw detection. https://github.com/GantMan/nsfw_model. [Online;

accessed 7-May-2023].

[99] https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words.

[Online; accessed 7-May-2023].

[100] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution

or region supervision. In ICML, 2021.

[101] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic,
and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images.
NeurIPS, 2022.

[102] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren

Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

[103] Edward Hu, Yelong Shen, Phil Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Lu Wang, and Weizhu Chen. Lora:

Low-rank adaptation of large language models, 2021.

[104] Jiaxiang

Stable-dreamfusion:
https://github.com/ashawkey/stable-dreamfusion.

Tang.

Text-to-3d with

stable-diffusion,

2022.

[105] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim,
Jiyoung Lee, and Seungryong Kim. Let 2d diffusion model know 3d-consistency for robust text-to-3d
generation. arXiv, 2023.

[106] Donghoon Lee, Jiseob Kim, Jisu Choi, Jongmin Kim, Minwoo Byeon, Woonhyuk Baek, and Saehoon

Kim. Karlo-v1.0.alpha on coyo-100m and cc15m. https://github.com/kakaobrain/karlo, 2022.

[107] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans

trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.

14

Appendix A Price Breakdown Details

This section provides our details computation for Table 1. Using a single A40 GPU, BLIP2 runs at
∼ 2700 iterations per hour, enabling it to process around ∼ 337.5 objects hourly given the eight-run
requirement for generating captions for 8 rendering views. This translates to about 2.96 hours to
process 1k objects, costing 2.96 × $1.28 = $3.79 with the rate $1.28/hr on the cloud platform,
CoreWeave. On the same A40 GPU, CLIP operates at ∼ 27000 iterations per hour, incurring a cost
of $0.38. Importantly, utilizing eight A40s costs the same as using one, due to the parallel processing
capacity across multiple GPUs for multiple rendering views.

We compute our GPT4 cost by averaging input token numbers, as OpenAI GPT4 API (8k context)
costs 0.03/1k tokens, Our input prompt is: “Given a set of descriptions about the same 3D object,
distill these descriptions into one concise caption. The descriptions are as follows: ‘captions’. Avoid
describing background, surface, and posture. The caption should be:", which consists of (1) text
prompt and (2) captions generated by BLIP2 or BLIP2 + CLIP. Without CLIP’s filtering, our input
prompt contains 40 captions which have ∼ 511.1 tokens on average, cost 511.1/1000×0.03×1000 =
$15.33 for 1k objects. With CLIP, our input prompt contains 8 captions which have ∼ 139.3 tokens
on average, cost 139.3/1000 × 0.03 × 1000 = $4.18 for 1k objects.

The average cost per 1k objects for human-authored annotation is computed as the average expenditure
on the crowdsourcing platform, Hive. The human annotation speed is computed by averaging the
annotation progress across our whole annotation process.

We do not report the average cost of Cap3D (QA) in the main paper, as we only use it on ABO. For
completeness, we report it here. The one distinction is BLIP2 is run twice instead of once for the
two-stage question answering (QA). The cost of BLIP2 thus doubles, from $3.79 to $7.58; and total
cost increases from $8.35 to $12.14 per 1k objects.

Appendix B Additional 3D Captioning Results

Figure 7: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

15

3D low poly model of a soldier/warrior with a sword and shield.3D model of a red and white molecule.A 3D model of a helmet with a red, yellow, and black gas respirator mask and a red light.A 3D model of a red and white mushroom.A 3D rendering of a spiral staircase with a railing in a white room with an open door.A red and blue leather suitcase with a cross, resembling an old medical bag or first aid box.zA 3D model of a Rubik's Cube featuring blue, orange, red, green, and yellow squares.A 3D model of a robotic horse with wings and spikes.3D model of a green and yellow metal truss with two holes and cross beam.A 3D model of a house with a red roof, fence, and carousel.A white 3D printed figurine of Santa Claus with reindeer antlers, holding a stick and standing on a rock.L-shaped sectional sofa with a chaise, U-shaped backrest, curved armrests, and a footstool on one side.Figure 8: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

Figure 9: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

16

UFC card featuring a shirtless man in red and white design.3D white submarine model with a gun and pole feature.3D Wooden Barrel ModelA 3D model of a stone mask sculpture resembling a broken piece of ice.A 3D ornate picture frame with a light bulb in it.3D rendering of a house with a roofz3D white satellite model with antennaA set of three colorful 3D model houses with roofs.3D model of Earth in a circle shape.A white 3D figurine of a bird on top of a shoe.Star Wars Boba Fett 3D Model3D model of a pink plastic chairA 3D model of a white robot.A 3D model of a green, armored lizard wearing a crown and holding a sword.A 3D model of a large white rock, possibly marble or granite, with a flag on it.A 3D model of a statue of a man with a hole in the ground.A 3D rendering of a row of vending machines and various colored boxes.A 3D model of a small flying robot-spaceship hybrid with extended arms, featuring an alien and a man on it.z3D wooden bear statue model3D printed kookaburra model sitting on a branch.A 3D yellow table with a cup of coffee on top.A 3D model of a white cylindrical object with features resembling a radiator, vase, and light bulb.3D illustration of a small yellow flower with leaves.3D model of Transformers Optimus Prime blue and red truckFigure 10: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

Figure 11: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

17

A 3D rendering of a house with a roof structure featuring pink lines.A white doily with a floral pattern and a circular ceiling light with a flower design.3D rendering of a white square object3D model of a wooden park bench with leaves on it.A 3D model of a modern blue leather sofa with metal legs and a gray metal shelf with two hooks.3D model of a green pea pod and propeller with two peas.z3D model of a marble nativity scene figurine.A 3D white skull model with red eyes, resembling a combination of animal, squid, and starfish features.3D model of a cowWhite Nintendo Wii console with a power outlet and USB port.A yellow gold ring with a pink sapphire stone.A 3D scene featuring a destroyed house, building, plane, and car, with a flying bird.3D model of a sword/daggerA white 3D model of a teddy bear with arms outstretched.A 3D model of a tall high-rise building.A 3D model of a robot with a table and satellite.3D model of a green and yellow box with a hole in it.A 3D model of a twisted, colorful object resembling a ball with purple, pink, and blue squiggles and wires.z3D model of a baseball batWhite Aston Martin DB5 3D ModelA 3D model featuring pillars, a fountain with statues, three vases, pots on a table, valves, a pipe, and light fixtures.A 3D model of a red shipping container with a blue and white logo and white label on it.3D model of a gravestone/tombstone.Brown bag with coins and a green string, featuring a coin falling out.Figure 12: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

Figure 13: Random 3D captioning examples generated by Cap3D. Two views of 3D objects (Obja-
verse [9]) are shown here, Cap3D uses eight.

18

A 3D model of a colorful, multi-colored boat with a flag, oars, and paddles.A purple and yellow toy lantern with a handle.A 3D model of a wooden door with a design, featuring a wooden sign, curved wall, and clock.A 3D model featuring a bird, twigs, fishing poles, a dragonfly, and skis with poles.A 3D blue box with a hole and sand on it.3D model of a small white and blue toy robot teddy bear.z3D model of a cricket bat, royalty-free vector illustration.A 3D rendering of a row of yellow poles.White 3D spiral staircase model3D model of a white axeWooden coat rack in a 3D model.A 3D model of a green plant with leaves and an eye.A 3D model of an armored hand wearing gloves with a leather strap and metal cuff.3D model of a white cloud.White plastic ring3D model of a small white building with stairs, featuring a cube and ceiling light fixture.3D model of a wooden power pole with wires.3D model of a white plastic bottle with a lid.zA 3D model of a small boat and house on a yellow platform.A knife and sliced bread on a wooden cutting board.A 3D model of a two-story house with a roof structure.A 3D rendering of a fish next to a Rubik's cube.A 3D model of a house on a grassy field with a road in front.3D model of a flower in a black box with a purple container and blue corner shelf.Figure 14: Comparative Analysis: Cap3D Generated Caption vs Human-Annotated Caption vs
Objaverse Metadata [9]. Two views of 3D objects are shown here, Cap3D and human use eight.

Figure 15: Comparative Analysis: Cap3D Generated Caption vs Human-Annotated Caption vs
Objaverse Metadata [9]. Two views of 3D objects are shown here, Cap3D and human use eight.

Appendix C Additional Text-to-3D Results

In this section, we provide several text-to-3D results for all of our compared methods. We include
Shap·E and Point·E pretrained models and the models finetuned on our data, as well as optimization
baselines, including DreamFusion, DreamField, and 3D Fuse.

19

3D model of a boy in a black outfit holding a sword with arms outstretched.3D model of a Qantas Airbus A380 airplane.A 3D model of black and white sneakers with a blue logo and a cat character, featuring white soles.White Nike Air Max Uptempo basketball sneaker 3D model.First Character in 3Ds max, Student Workbong 787Adidas scanned by Thunk3D handheld scanner, sampling at ratio 40%.  For more information, please kindly check as below, Whatsapp:...a cartoon kid with clothes black in 3da cartoon white commercial airplane in 3da grey shoe with a white frontgrey coloured sports shoe with undulations on its surface.Cap3DHumanMetadata3D model of a jar with a green lid.3D rendering of grey Champion sweatpants with red and black logo.A 3D model of a rusty, old train engine.This is a backup of a Poly Asset named Jar of jam. Saved from Poly by Google. Preview may be without textures, they are still in the Download ZIP with a preview thumbnail.a three layer structure with green oval top and white middle part and also having a brown base.a 3d model of a train old engine.a carton made antelope with horns horns on top of each otherCap3DHumanMetadataOld industrial diesel locomotive from Hungary.A 3D model of a white deer.a cartoon white pantsGame resolution Jogger pants made in the likeness of a pair of Joggers made by Champion. I started in Zbrush with the high res and exported to Maya for quad drawing the game res.. The model was also UV mapped and textured by myself. The texture was baked and finalized in Substance Painter. The poly count is just over 5200 tri's to fit the topology of the high res that was decimated from approximately 2,400,000 to around 415,000 quads.Figure 16: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

Figure 17: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

Figure 18: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

20

FinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)Dream FusionDream Field3D FuseReferenceA 3D model of a green teapot with horns.FinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)3D FuseDream FusionDream FieldReferenceA red and black table lamp with a black shade.FinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)3D FuseDream FusionDream FieldA 3D model of an axe featuring a dragon head, a sword, and a long, colorful handle.ReferenceFigure 19: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

Figure 20: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

Figure 21: Text-to-3D results. The top text prompt and “Reference" are from our test set. We
fine-tune the left 5-column methods on Cap3D-generated captions. The detailed setting and methods
are described in §5.3.

21

FinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)3D FuseDream FusionDream Field3D model of a Five Nights at Freddy's fox character with outstretched arms, wearing a hat and holding a gun.ReferenceFinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)3D FuseDream FusionDream FieldA 3D model of a mountain range with green grass and mountainous terrain.ReferenceFinetunedShap·E NeRFShap·E STFPretrainedPoint·E (Text-to-3D)S.Diff(LoRA) +Point·E (Im-to-3D)S.Diff(CNet) + Point·E (Im-to-3D)Dream FusionDream Field3D FuseReference3D model of witch hats and lanterns hanging from a chain.Appendix D Limitations and Failure Cases

As described in §3, Cap3D consists of four steps: (1) 3D objects rendering; (2) captioning via BLIP2;
(3) filtering captions via CLIP; (4) consolidate multiview information via GPT4. To effectively
capture comprehensive information through 2D renderings, our cameras are positioned above or
below the object. However, this sometimes leads to unusual 2D views, which cause the BLIP2 to
produce inaccurate information that CLIP cannot filter. Consequently, GPT4 struggles to consolidate
disparate information from multiple views, leading to ambiguous, verbose, and imprecise descriptions.
One example is shown in Figure 22. Moreover, our system struggles to accurately process certain
indoor 3D scans due to their inherent complexity (as shown in Figure 23), making them challenging
to distinguish, sometimes even for humans.

Note that, none of a caption from a single view can well describe the complete details from the given
3D object.

Figure 22: An failed case. The caption under each rendered image are generated by BLIP2 + filtered
by CLIP. The inaccurate content are highlighted with colors. GPT4 + CLIP cannot fix the error
generated by BLIP2 and result in a fuzzy description.

Figure 23: An failed case. The caption under each rendered image are generated by BLIP2 + filtered
by CLIP. The inaccurate content are highlighted with colors. The various views contain inaccurate
information. The associated details, roughly described, fail to accurately depict the indoor scene.

22

a 3d model of a dragon on a rocka 3d model of a butterfly on a rocka 3d model of a rock formation with a dragon on ita 3d model of a blue dragon on a rocka 3d model of a rock with a blue and yellow blob on ita 3d model of a dragon with rocks and watera 3d model of a rock with a girl sitting on ita 3d rendering of a rock with flowers on itGPT4Prompt: Given a set of descriptions about the same 3D object … distill these descriptions into one concise caption:A 3D model of a blue dragon on a rock formation with surrounding elements like a butterfly, girl, flowers, and water.Output Captiona 3d model of a house with a hole in ita model of a dump truck on a gray backgrounda 3d model of a house with a windowa 3d model of a boat in the middle of a field, a 3d model of a mud housea 3d model of a wooden boata 3d model of a rusted car on a gray backgrounda 3d model of a torn piece of woodGPT4Prompt: Given a set of descriptions about the same 3D object … distill these descriptions into one concise caption:A 3D model featuring a house with a hole and window, a boat in a field, a mud house, a dump truck, a wooden boat, and a rusted car.Output CaptionAppendix E ABO Captioning: Automated Metrics

In §5.2, we report human A/B judgments on ABO. We do not report automated metrics, which
are poor measures of performance for at least two reasons. First, ABO contains a large number
of objects that are very similar, meaning it would be challenging for captions to distinguish their
differences. Thus, retrieval metrics such as ViLT Image or Text Retrieval will show very poor scores
across metrics. Second, we show automated captioning performs poorly at describing geometry well,
meaning it is likely automated image-caption alignment will not align based on geometry well. For
completeness, we report automated metrics in Table 8. As expected, all retrieval scores are very low.
Automated captioning scores best across automated metrics, however we caution against drawing
conclusions from this result. Human studies in Table 4 suggest the opposite, and qualitative results
agree with this finding, e.g. Figure 4.

Table 8: ABO Automated Caption Evaluations. Automated captions are a poor measure of
performance on ABO as (1) many objects are similar, making retrieval difficult; (2) automated
captioning does not describe geometry well, so we should not expect automated image-caption
alignment to describe geometrically correct captions well.

Method

CLIP ViLT Img Retr. ViLT Text Retr.
R@10
Score R@5 R@10 R@5

Meta
Human
Cap3D
Cap3D(QA)

61.9
75.2
89.9
82.7

0.8
2.6
4.2
2.9

1.7
4.4
7.2
5.3

0.8
2.3
3.2
2.4

1.7
4.2
5.6
4.3

In contrast with A/B tests, which take place on the full 6.4k objects of ABO, this table is computed
on a random 5k object subset of ABO to follow standard retrieval benchmarks (performance drops
considerably as dataset size increases. Using 5k instead of the full 6.4k makes it much easier to
contextualize retrieval numbers). A/B performance on this 5k subset is very close to the full 6.4k
dataset, meaning the sample is highly representative, and one can compare the results from this table
in combination with Table 4 in the main paper.

Appendix F Additional Details

F.1 Prompt used in Cap3D

The two prompts used for BLIP2 used in Cap3D (QA) are (1) “Question: what object is in this image?
Answer:" and (2) “Question: what is the structure and geometry of this <object>?" where <object> is
replaced with the response to prompt (1).

For the prompt used in GPT4, we used “Given a set of descriptions about the same 3D object, distill
these descriptions into one concise caption. The descriptions are as follows: ’captions’. Avoid
describing background, surface, and posture. The caption should be:". We did several prompt
engineering and considered prompt with more context, like “Below you will find a set of descriptions,
each one is originating from various renderings of an identical 3D object. The level of accuracy in
these descriptions ranges significantly: some might not correspond to the 3D object at all, others
could be entirely accurate, while a few may only partially represent the object. Your task involves
scrutinizing these descriptions and distilling them into a single, holistic depiction. The descriptions
are as follows: ‘captions’. Note: Please avoid using the phrases ’grey background’, ’gray background’,
and ’gray surface’ in your consolidated depiction. The synthesized description of the 3D object
should be:". However, with those longer prompt with more context, we noticed GPT4 sometimes
would generate its reasoning process which led to confusing output captions. Also, for the sake of
cost, we hope to make our prompt as short as possible.

F.2 Rendering Details

We use Blender to render 3D objects in Objaverse [9] and ABO [35]. For each object, we first
normalize them into a unit cube and recenter to origin. Then, we place 8 different cameras surrounding

23

the object with 2 cameras slightly below the object to capture the bottom of the object. Three area
lights are placed and function as key light, fill light, and rim light, respectively. The detailed
parameters are listed in our rendering script, provided in our Github.

According to § 3.2, we filter objects in Objaverse based on commerical-license, rendering information,
and ethicial standards, and results a subset of 660k objects for rendering and captioning. In ABO,
we exclude categories with simple geometry to concentrate on geometrical captioning, including
“BLANKET", “RUG", “WALL_ART", “PLACEMAT", “CURTAIN", “MOUSE_PAD". This resulting
a final subset of 6.4k objects for rendering and captioning.

F.3 Human Captioning Split

Human captions are collected on a manually selected subset of Objaverse with good renders of
nontrivial but decipherable objects. These objects are likely to be the most sensible for captioning
and A/B testing. For instance, some Objaverse objects are essentially a simple rock with little texture;
in others it can be difficult for a human to describe an object (e.g. abstract art, no clear object visible,
or 3D scans with hard-to-distinguish details). These excluded objects are generally not effective
samples to use for human A/B testing, as the correct caption may not be clear or may be trivial. We
also exclude furniture, which is suitable for captioning, but we measure this with more focus on ABO.
Human captions on ABO follow the split of [77].

Appendix G Crowdsourced Captioning Details

We use Hive for crowdsourced captioning. Workers are given instructions for the task including
gold-standard examples. Captioning instructions are shared below for Objaverse in Figure 24 and
ABO in Figure 25. Workers are persistently monitored. If a worker produces bad captions they
are promptly banned from captioning, and their previous captions are discarded. Workers are paid
approximately $50 per 1k tasks. We do not have access to their captioning rates; assuming a rate of 3
objects per minute, this would result in $9 per hour. Across Objaverse and ABO we spend a total of
$7k on captioning.

24

Figure 25: ABO Caption Instructions.

25

Figure 24: Objaverse Caption Instructions.

26

Appendix H Crowdsourced A/B Testing Details

We use Hive for crowdsourced A/B testing. Specifically, workers are given an image and two captions,
and select which is better on a scale from 1 to 5, where 3 is a tie. So 1 would be "left much better",
and 2 would be "left better". Workers are given instructions for the task along with gold standard
examples. Workers are informed to prioritize accuracy, then informative detail, then brevity. Left/right
order between methods was randomized for each instance. A/B Testing instructions are shared below
for Objaverse in Figure 27 and ABO in Figure 26.

Workers are automatically banned by the platform if they miss too many gold-standard examples.
However, we found some workers would successfully pass the handful of gold-standard examples
while scamming on the rest of the examples. The most common scam cases were always picking the
same number, or always picking the shorter or longer caption. We thus manually search through all
workers and ban workers who meet these scamming criteria and discard their judgments. Unfortu-
nately, discarding judgments leads to uneven numbers of observations for each individual experiment.
Nevertheless, in all cases, enough observations are available to draw conclusive findings.

The size of each experiment’s data after discarded judgments is below.

• Objaverse Split (1) takes place on a random set upon which human captions are available.

Cap3D vs. Human has 36k observations across 22k objects.

• Objaverse Split (2) takes place on a random object set upon which human captions are
available. Cap3D vs. Human has 10k observations across 4.7k objects. Cap3D vs. Metadata
has 7k observations across 4.7k objects (less than the target 10k), though given the extremely
poor rating of Metadata, results are conclusive.

• Objaverse Split (3) takes place on a random object set upon the entire Objaverse dataset.
Cap3D vs. BLIP2 has 20k observations across 5.0k objects and Cap3D vs. +GPT4 has 29k
observations across 5.0k objects.

• ABO takes place on the full ABO object set. Human vs. Cap3D has 21k observations across
6.4k objects, Cap3D (QA) vs. Human has 17k observations across 6.4k objects, Cap3D
(QA) vs. Cap3D has 13k observations across 6.4k objects, and Cap3D (QA) vs. Meta has
12k observations across 6.4k objects.

Workers are paid approximately $20 per 1k tasks. We do not have access to their captioning rates;
assuming a rate of 7.5 A/B tests selected per minute, this would result in $9 per hour. Across
Objaverse and ABO we spent a total of $1.8k on A/B testing.

27

Figure 26: A/B Instructions: Objaverse Captions.

28

Figure 27: A/B Instructions: ABO Captions.

29

Appendix I Additional Experimental Details

Captioning: we perform one full-scale evaluation run for all captioning experiments; 95% confidence
interval for mean is presented. Metrics are overviewed in §5.1; A/B testing is detailed further in §H.
CLIP Score takes about 5 minutes, while ViLT R-Precision takes about 8 hours using an A40 for test
set of 5k object-caption pairs. Crowdsourced A/B testing takes about 12 hours for 10k responses
across 5k objects.

Text-to-3D, finetuning: for finetuning experiments, we used one train and evaluation run using a
learning rate validated on a small overfitting experiment on the train set. Training took about 3 days on
the full set and 1 day on the small (human) set. We used AdamW optimizer and CosineAnnealingLR
scheduler with initial learning rate 1e − 5 for finetuning both Point·E and Shap·E. We adopted batch
size 64 and 256 for Shap·E and Point·E, respectively. However, for Shap·E, we found it usually
outputs NaN and needed to re-start from saved checkpoints, which could be one of the reaons why our
finetune did not bring improvements. For LoRA, we use AdamW optimizer and CosineAnnealingLR
scheduler with initial learning rate 1e − 4 and batch size of 3. For ControlNet, we use AdamW
optimizer and constant learning rate of 1e − 5 and batch size of 8. Experiments use 4 A40s to train
except LoRA, which fails upon multi-gpu training due to a HuggingFace internal DDP error. Notably
single-gpu training still yields improvement. Evaluation takes the following time (in seconds) per
iteration, which includes rendering:

• PointE (text-to-3D): 37sec = 28sec (text-to-3D) + 9sec (render)
• LoRA + PointE(im-to-3D): 114sec = 5sec + 100sec (im-to-3D) + 9sec (render)
• ControlNet + PointE(im-to-3D): 124sec = 15sec + 100sec (im-to-3D) + 9sec (render)
• ShapE (NeRF): 193sec (text-to-3D + render)
• ShapE (stf): 16sec (text-to-3D + render)

Note publicly available PointE (im-to-3D) is 1B param, making it slower than the largest publicly
available PointE (text-to-3D) of 40M. Evaluation metrics are detailed in §5.3.

Text-to-3D, optimization: For one object, optimization plus final rendering takes 40 minutes for
3DFuse, 95 minutes for Stable DreamFusion, and 35 minutes for DreamField; using 1 A40 GPU. We
use default parameters for all methods and run them once.

30


