FiLM: Visual Reasoning with a General Conditioning Layer

Ethan Perez1,2, Florian Strub4, Harm de Vries1, Vincent Dumoulin1, Aaron Courville1,3
1MILA, Universit´e de Montr´eal, 2Rice University, 3CIFAR Fellow,
4Univ. Lille, CNRS, Centrale Lille, Inria, UMR 9189 CRIStAL France
ethanperez@rice.edu, ﬂorian.strub@inria.fr, mail@harmdevries.com,{dumouliv,courvila}@iro.umontreal.ca

7
1
0
2
c
e
D
8
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
1
7
8
7
0
.
9
0
7
1
:
v
i
X
r
a

Abstract

We introduce a general-purpose conditioning method for neu-
ral networks called FiLM: Feature-wise Linear Modulation.
FiLM layers inﬂuence neural network computation via a sim-
ple, feature-wise afﬁne transformation based on conditioning
information. We show that FiLM layers are highly effective
for visual reasoning — answering image-related questions
which require a multi-step, high-level process — a task which
has proven difﬁcult for standard deep learning methods that
do not explicitly model reasoning. Speciﬁcally, we show on
visual reasoning tasks that FiLM layers 1) halve state-of-the-
art error for the CLEVR benchmark, 2) modulate features in
a coherent manner, 3) are robust to ablations and architectural
modiﬁcations, and 4) generalize well to challenging, new data
from few examples or even zero-shot.

1

Introduction

The ability to reason about everyday visual input is a fun-
damental building block of human intelligence. Some have
argued that for artiﬁcial agents to learn this complex, struc-
tured process, it is necessary to build in aspects of reason-
ing, such as compositionality (Hu et al. 2017; Johnson et
al. 2017b) or relational computation (Santoro et al. 2017).
However, if a model made from general-purpose compo-
nents could learn to visually reason, such an architecture
would likely be more widely applicable across domains.

To understand if such a general-purpose architecture ex-
ists, we take advantage of the recently proposed CLEVR
dataset (Johnson et al. 2017a) that tests visual reasoning via
question answering. Examples from CLEVR are shown in
Figure 1. Visual question answering, the general task of ask-
ing questions about images, has its own line of datasets (Ma-
linowski and Fritz 2014; Geman et al. 2015; Antol et al.
2015) which generally focus on asking a diverse set of
simpler questions on images, often answerable in a single
glance. From these datasets, a number of effective, general-
purpose deep learning models have emerged for visual ques-
tion answering (Malinowski, Rohrbach, and Fritz 2015;
Yang et al. 2016; Lu et al. 2016; Anderson et al. 2017). How-
ever, tests on CLEVR show that these general deep learning
approaches struggle to learn structured, multi-step reason-
ing (Johnson et al. 2017a). In particular, these methods tend

Copyright c(cid:13) 2018, Association for the Advancement of Artiﬁcial
Intelligence (www.aaai.org). All rights reserved.

(a) Q: What number of
cylinders are small pur-
ple things or yellow rubber
things? A: 2

(b) Q: What color is the
other object that is the same
shape as the large brown
matte thing? A: Brown

Figure 1: CLEVR examples and FiLM model answers.

to exploit biases in the data rather than capture complex un-
derlying structure behind reasoning (Goyal et al. 2017).

In this work, we show that a general model architecture
can achieve strong visual reasoning with a method we intro-
duce as FiLM: Feature-wise Linear Modulation. A FiLM
layer carries out a simple, feature-wise afﬁne transformation
on a neural network’s intermediate features, conditioned on
an arbitrary input. In the case of visual reasoning, FiLM lay-
ers enable a Recurrent Neural Network (RNN) over an input
question to inﬂuence Convolutional Neural Network (CNN)
computation over an image. This process adaptively and rad-
ically alters the CNN’s behavior as a function of the input
question, allowing the overall model to carry out a variety
of reasoning tasks, ranging from counting to comparing, for
example. FiLM can be thought of as a generalization of Con-
ditional Normalization, which has proven highly successful
for image stylization (Dumoulin, Shlens, and Kudlur 2017;
Ghiasi et al. 2017; Huang and Belongie 2017), speech recog-
nition (Kim, Song, and Bengio 2017), and visual question
answering (de Vries et al. 2017), demonstrating FiLM’s
broad applicability.

In this paper, which expands upon a shorter report (Perez
et al. 2017), our key contribution is that we show FiLM is
a strong conditioning method by showing the following on
visual reasoning tasks:

1. FiLM models achieve state-of-the-art across a variety of
visual reasoning tasks, often by signiﬁcant margins.

2. FiLM operates in a coherent manner. It learns a complex,
underlying structure and manipulates the conditioned net-
work’s features in a selective manner. It also enables the

CNN to properly localize question-referenced objects.

3. FiLM is robust; many FiLM model ablations still outper-
form prior state-of-the-art. Notably, we ﬁnd there is no
close link between normalization and the success of a con-
ditioned afﬁne transformation, a previously untouched as-
sumption. Thus, we relax the conditions under which this
method can be applied.

4. FiLM models learn from little data to generalize to more
complex and/or substantially different data than seen dur-
ing training. We also introduce a novel FiLM-based zero-
shot generalization method that further improves and val-
idates FiLM’s generalization capabilities.

2 Method

Our model processes the question-image input using FiLM,
illustrated in Figure 2. We start by explaining FiLM and then
describe our particular model for visual reasoning.

2.1 Feature-wise Linear Modulation

FiLM learns to adaptively inﬂuence the output of a neural
network by applying an afﬁne transformation, or FiLM, to
the network’s intermediate features, based on some input.
More formally, FiLM learns functions f and h which output
γi,c and βi,c as a function of input xi:

γi,c = fc(xi)

βi,c = hc(xi),

(1)

where γi,c and βi,c modulate a neural network’s activations
Fi,c, whose subscripts refer to the ith input’s cth feature or
feature map, via a feature-wise afﬁne transformation:

F iLM (Fi,c|γi,c, βi,c) = γi,cFi,c + βi,c.

(2)

f and h can be arbitrary functions such as neural networks.
Modulation of a target neural network’s processing can be
based on the same input to that neural network or some other
input, as in the case of multi-modal or conditional tasks. For
CNNs, f and h thus modulate the per-feature-map distribu-
tion of activations based on xi, agnostic to spatial location.
In practice, it is easier to refer to f and h as a single func-
tion that outputs one (γ, β) vector, since, for example, it
is often beneﬁcial to share parameters across f and h for
more efﬁcient learning. We refer to this single function as
the FiLM generator. We also refer to the network to which
FiLM layers are applied as the Feature-wise Linearly Mod-
ulated network, the FiLM-ed network.

FiLM layers empower the FiLM generator to manipulate
feature maps of a target, FiLM-ed network by scaling them
up or down, negating them, shutting them off, selectively
thresholding them (when followed by a ReLU), and more.
Each feature map is conditioned independently, giving the
FiLM generator moderately ﬁne-grained control over acti-
vations at each FiLM layer.

As FiLM only requires two parameters per modulated fea-
ture map, it is a scalable and computationally efﬁcient con-
ditioning method. In particular, FiLM has a computational
cost that does not scale with the image resolution.

Figure 2: A single FiLM layer for a CNN. The dot signiﬁes
a Hadamard product. Various combinations of γ and β can
modulate individual feature maps in a variety of ways.

2.2 Model
Our FiLM model consists of a FiLM-generating linguis-
tic pipeline and a FiLM-ed visual pipeline as depicted in
Figure 3. The FiLM generator processes a question xi us-
ing a Gated Recurrent Unit (GRU) network (Chung et al.
2014) with 4096 hidden units that takes in learned, 200-
dimensional word embeddings. The ﬁnal GRU hidden state
is a question embedding, from which the model predicts
i,·, βn
i,·) for each nth residual block via afﬁne projection.
(γn
The visual pipeline extracts 128 14 × 14 image feature
maps from a resized, 224 × 224 image input using either
a CNN trained from scratch or a ﬁxed, pre-trained feature
extractor with a learned layer of 3 × 3 convolutions. The
CNN trained from scratch consists of 4 layers with 128 4 ×
4 kernels each, ReLU activations, and batch normalization,
similar to prior work on CLEVR (Santoro et al. 2017). The
ﬁxed feature extractor outputs the conv4 layer of a ResNet-
101 (He et al. 2016) pre-trained on ImageNet (Russakovsky
et al. 2015) to match prior work on CLEVR (Johnson et al.
2017a; 2017b). Image features are processed by several —
4 for our model — FiLM-ed residual blocks (ResBlocks)
with 128 feature maps and a ﬁnal classiﬁer. The classiﬁer
consists of a 1 × 1 convolution to 512 feature maps, global
max-pooling, and a two-layer MLP with 1024 hidden units
that outputs a softmax distribution over ﬁnal answers.

Each FiLM-ed ResBlock starts with a 1 × 1 convolu-
tion followed by one 3 × 3 convolution with an architec-
ture as depicted in Figure 3. We turn the parameters of batch
normalization layers that immediately precede FiLM layers
off. Drawing from prior work on CLEVR (Hu et al. 2017;
Santoro et al. 2017) and visual reasoning (Watters et al.
2017), we concatenate two coordinate feature maps indi-
cating relative x and y spatial position (scaled from −1 to
1) with the image features, each ResBlock’s input, and the
classiﬁer’s input to facilitate spatial reasoning.

We train our model end-to-end from scratch with

Notably, prior work in CN has not examined whether
the afﬁne transformation must be placed directly after nor-
malization. Rather, prior work includes normalization in the
method name for instructive purposes or due to implemen-
tation details. We investigate the connection between FiLM
and normalization, ﬁnding it not strictly necessary for the
afﬁne transformation to occur directly after normalization.
Thus, we provide a uniﬁed framework for all of these meth-
ods through FiLM, as well as a normalization-free relaxation
of this approach which can be more broadly applied.

Beyond CN, there are many connections between FiLM
and other conditioning methods. A common approach, used
for example in Conditional DCGANs (Radford, Metz, and
Chintala 2016), is to concatenate constant feature maps
of conditioning information with convolutional layer input.
Though not as parameter efﬁcient, this method simply re-
sults in a feature-wise conditional bias. Likewise, concate-
nating conditioning information with fully-connected layer
input amounts to a feature-wise conditional bias. Other ap-
proaches such as WaveNet (van den Oord et al. 2016a) and
Conditional PixelCNN (van den Oord et al. 2016b) directly
add a conditional feature-wise bias. These approaches are
equivalent to FiLM with γ = 1, which we compare FiLM
to in the Experiments section. In reinforcement learning,
an alternate formulation of FiLM has been used to train
one game-conditioned deep Q-network to play ten Atari
games (Kirkpatrick et al. 2017), though FiLM was neither
the focus of this work nor analyzed as a major component.

Other methods gate an input’s features as a function of
that same input, rather than a separate conditioning in-
put. These methods include LSTMs for sequence model-
ing (Hochreiter and Schmidhuber 1997), Convolutional Se-
quence to Sequence for machine translation (Gehring et al.
2017), and even the ImageNet 2017 winning model, Squeeze
and Excitation Networks (Hu, Shen, and Sun 2017). This
approach amounts to a feature-wise, conditional scaling, re-
stricted to between 0 and 1, while FiLM consists of both
scaling and shifting, each unrestricted. In the Experiments
section, we show the effect of restricting FiLM’s scaling to
between 0 and 1 for visual reasoning. We ﬁnd it noteworthy
that this general approach of feature modulation is effective
across a variety of settings and architectures.

There are even broader links between FiLM and other
methods. For example, FiLM can be viewed as using one
network to generate parameters of another network, mak-
ing it a form of hypernetwork (Ha, Dai, and Le 2016). Also,
FiLM has potential ties with conditional computation and
mixture of experts methods, where specialized network sub-
parts are active on a per-example basis (Jordan and Jacobs
1994; Eigen, Ranzato, and Sutskever 2014; Shazeer et al.
2017); we later provide evidence that FiLM learns to selec-
tively highlight or suppress feature maps based on condi-
tioning information. Those methods select at a sub-network
level while FiLM selects at a feature map level.

In the domain of visual reasoning, one leading method is
the Program Generator + Execution Engine model (John-
son et al. 2017b). This approach consists of a sequence-
to-sequence Program Generator, which takes in a question
and outputs a sequence corresponding to a tree of compos-

Figure 3: The FiLM generator (left), FiLM-ed network (mid-
dle), and residual block architecture (right) of our model.

Adam (Kingma and Ba 2015) (learning rate 3e−4), weight
decay (1e−5), batch size 64, and batch normalization and
ReLU throughout FiLM-ed network. Our model uses only
image-question-answer triplets from the training set with-
out data augmentation. We employ early stopping based on
validation accuracy, training for 80 epochs maximum. Fur-
ther model details are in the appendix. Empirically, we found
FiLM had a large capacity, so many architectural and hyper-
parameter choices were for added regularization.

We stress that our model relies solely on feature-wise
afﬁne conditioning to use question information inﬂuence the
visual pipeline behavior to answer questions. This approach
differs from classical visual question answering pipelines
which fuse image and language information into a single
embedding via element-wise product, concatenation, atten-
tion, and/or more advanced methods (Yang et al. 2016;
Lu et al. 2016; Anderson et al. 2017).

3 Related Work
FiLM can be viewed as a generalization of Conditional Nor-
malization (CN) methods. CN replaces the parameters of the
feature-wise afﬁne transformation typical in normalization
layers, as introduced originally (Ioffe and Szegedy 2015),
with a learned function of some conditioning information.
Various forms of CN have proven highly effective across a
number of domains: Conditional Instance Norm (Dumoulin,
Shlens, and Kudlur 2017; Ghiasi et al. 2017) and Adaptive
Instance Norm (Huang and Belongie 2017) for image styl-
ization, Dynamic Layer Norm for speech recognition (Kim,
Song, and Bengio 2017), and Conditional Batch Norm for
general visual question answering on complex scenes such
as VQA and GuessWhat?! (de Vries et al. 2017). This work
complements our own, as we seek to show that feature-wise
afﬁne conditioning is effective for multi-step reasoning and
understand the underlying mechanism behind its success.

Model

Overall

Count

Exist

Compare
Numbers

Query
Attribute

Compare
Attribute

Human (Johnson et al. 2017b)

Q-type baseline (Johnson et al. 2017b)
LSTM (Johnson et al. 2017b)
CNN+LSTM (Johnson et al. 2017b)
CNN+LSTM+SA (Santoro et al. 2017)
N2NMN* (Hu et al. 2017)
PG+EE (9K prog.)* (Johnson et al. 2017b)
PG+EE (700K prog.)* (Johnson et al. 2017b)
CNN+LSTM+RN†‡ (Santoro et al. 2017)

CNN+GRU+FiLM
CNN+GRU+FiLM‡

92.6

41.8
46.8
52.3
76.6
83.7
88.6
96.9
95.5

97.7
97.6

86.7

34.6
41.7
43.7
64.4
68.5
79.7
92.7
90.1

94.3
94.3

96.6

50.2
61.1
65.2
82.7
85.7
89.7
97.1
97.8

99.1
99.3

86.5

51.0
69.8
67.1
77.4
84.9
79.1
98.7
93.6

96.8
93.4

95.0

36.0
36.8
49.3
82.6
90.0
92.6
98.1
97.9

99.1
99.3

96.0

51.3
51.8
53.0
75.4
88.7
96.0
98.9
97.1

99.1
99.3

Table 1: CLEVR accuracy (overall and per-question-type) by baselines, competing methods, and FiLM. (*) denotes use of
extra supervision via program labels. (†) denotes use of data augmentation. (‡) denotes training from raw pixels.

able neural modules, each of which is a two or three layer
residual block. This tree of neural modules is assembled to
form the Execution Engine that then predicts an answer from
the image. This modular approach is part of a line of neu-
ral module network methods (Andreas et al. 2016a; 2016b;
Hu et al. 2017), of which End-to-End Module Networks (Hu
et al. 2017) have also been tested on visual reasoning. These
models use strong priors by explicitly modeling the compo-
sitional nature of reasoning and by training with additional
program labels, i.e. ground-truth step-by-step instructions
on how to correctly answer a question. End-to-End Mod-
ule Networks further build in model biases via per-module,
hand-crafted neural architectures for speciﬁc functions. Our
approach learns directly from visual and textual input with-
out additional cues or a specialized architecture.

Relation Networks (RNs) are another leading approach
for visual reasoning (Santoro et al. 2017). RNs succeed by
explicitly building in a comparison-based prior. RNs use an
MLP to carry out pairwise comparisons over each location
of extracted convolutional features over an image, includ-
ing LSTM-extracted question features as input to this MLP.
RNs then element-wise sum over the resulting comparison
vectors to form another vector from which a ﬁnal classi-
ﬁer predicts the answer. We note that RNs have a compu-
tational cost that scales quadratically in spatial resolution,
while FiLM’s cost is independent of spatial resolution. No-
tably, since RNs concatenate question features with MLP in-
put, a form of feature-wise conditional biasing as explained
earlier, their conditioning approach is related to FiLM.

4 Experiments
First, we test our model on visual reasoning with the CLEVR
task and use trained FiLM models to analyze what FiLM
learns. Second, we explore how well our model generalizes
to more challenging questions with the CLEVR-Humans
task. Finally, we examine how FiLM performs in few-
shot and zero-shot generalization settings using the CLEVR
Compositional Generalization Test. In the appendix, we pro-
vide an error analysis of our model. Our code is available
at https://github.com/ethanjperez/film.

4.1 CLEVR Task
CLEVR is a synthetic dataset of 700K (image, question, an-
swer, program) tuples (Johnson et al. 2017a). Images con-
tain 3D-rendered objects of various shapes, materials, col-
ors, and sizes. Questions are multi-step and compositional
in nature, as shown in Figure 1. They range from counting
questions (“How many green objects have the same size as
the green metallic block?”) to comparison questions (“Are
there fewer tiny yellow cylinders than yellow metal cubes?”)
and can be 40+ words long. Answers are each one word from
a set of 28 possible answers. Programs are an additional
supervisory signal consisting of step-by-step instructions,
such as filter shape[cube], relate[right], and
count, on how to answer the question.

Baselines We compare against the following methods, dis-
cussed in detail in the Related Work section:
• Q-type baseline: Predicts based on a question’s category.
• LSTM: Predicts using only the question.
• CNN+LSTM: MLP prediction over CNN-extracted im-

age features and LSTM-extracted question features.

• Stacked Attention Networks (CNN+LSTM+SA): Lin-
ear prediction over CNN-extracted image feature and
LSTM-extracted question features combined via two
rounds of soft spatial attention (Yang et al. 2016).

• End-to-End Module Networks (N2NMN) and Pro-
gram Generator + Execution Engine (PG+EE): Meth-
ods in which separate neural networks learn separate sub-
functions and are assembled into a question-dependent
structure (Hu et al. 2017; Johnson et al. 2017b).

• Relation Networks (CNN+LSTM+RN): An approach
which builds in pairwise comparisons over spatial lo-
cations to explicitly model reasoning’s relational na-
ture (Santoro et al. 2017).

Results FiLM achieves a new overall state-of-the-art on
CLEVR, as shown in Table 1, outperforming humans and
previous methods, including those using explicit models of
reasoning, program supervision, and/or data augmentation.

Q: What shape is the...

...purple thing? A: cube

...blue thing? A: sphere

...red thing right of the
blue thing? A: sphere

...red thing left of
blue thing? A: cube

the

Q: How many
things are...

cyan

...right of the gray cube?
A: 3

...left of the small cube?
A: 2

...right of the gray cube
and left of
the small
cube? A: 1

...right of the gray cube
or left of the small cube?
A: 4 (P: 3)

Figure 4: Visualizations of the distribution of locations which the model uses for its globally max-pooled features which its ﬁnal
MLP predicts from. FiLM correctly localizes the answer-referenced object (top) or all question-referenced objects (bottom),
but not as accurately when it answers incorrectly (rightmost bottom). Questions and images used match (Johnson et al. 2017b).

For methods not using extra supervision, FiLM roughly
halves state-of-the-art error (from 4.5% to 2.3%). Note that
using pre-trained image features as input can be viewed as a
form of data augmentation in itself but that FiLM performs
equally well using raw pixel inputs. Interestingly, the raw
pixel model seems to perform better on lower-level ques-
tions (i.e. querying and comparing attributes) while the im-
age features model seems to perform better on higher-level
questions (i.e. compare numbers of objects).

4.2 What Do FiLM Layers Learn?
To understand how FiLM visually reasons, we visualize acti-
vations to observe the net result of FiLM layers. We also use
histograms and t-SNE (van der Maaten and Hinton 2008) to
ﬁnd patterns in the learned FiLM γ and β parameters them-
selves. In Figures 14 and 15 in the appendix, we visualize
the effect of FiLM at the single feature map level.

Activation Visualizations Figure 4 visualizes the distri-
bution of locations responsible for the globally-pooled fea-
tures which the MLP in the model’s ﬁnal classiﬁer uses
to predict answers. These images reveal that the FiLM
model predicts using features of areas near answer-related
or question-related objects, as the high CLEVR accuracy
also suggests. This ﬁnding highlights that appropriate fea-
ture modulation indirectly results in spatial modulation, as
regions with question-relevant features will have large acti-
vations while other regions will not. This observation might
explain why FiLM outperforms Stacked Attention, the next
best method not explicitly built for reasoning, so signiﬁ-
cantly (21%); FiLM appears to carry many of spatial atten-
tion’s beneﬁts, while also inﬂuencing feature representation.
Figure 4 also suggests that the FiLM-ed network carries
out reasoning throughout its pipeline. In the top example, the
FiLM-ed network has localized the answer-referenced ob-
ject alone before the MLP classiﬁer. In the bottom example,
the FiLM-ed network retains, for the MLP classiﬁer, fea-

Figure 5: Histograms of γi,c (left) and βi,c (right) values
over all FiLM layers, calculated over the validation set.

tures on objects that are not referred to by the answer but are
referred to by the question. The latter example provides ev-
idence that the ﬁnal MLP itself carries out some reasoning,
using FiLM to extract relevant features for its reasoning.

FiLM Parameter Histograms To analyze at a lower level
how FiLM uses the question to condition the visual pipeline,
we plot γ and β values predicted over the validation set, as
shown in Figure 5 and in more detail in the appendix (Fig-
ures 16 to 18). γ and β values take advantage of a sizable
range, varying from -15 to 19 and from -9 to 16, respec-
tively. γ values show a sharp peak at 0, showing that FiLM
learns to use the question to shut off or signiﬁcantly sup-
press whole feature maps. Simultaneously, FiLM learns to
upregulate a much more selective set of other feature maps
with high magnitude γ values. Furthermore, a large frac-
tion (36%) of γ values are negative; since our model uses
a ReLU after FiLM, γ < 0 can cause a signiﬁcantly differ-
ent set of activations to pass the ReLU to downstream layers
than γ > 0. Also, 76% of β values are negative, suggest-
ing that FiLM also uses β to be selective about which acti-
vations pass the ReLU. We show later that FiLM’s success
is largely architecture-agnostic, but examining a particular
model gives insight into the inﬂuence FiLM learns to exert

Figure 6: t-SNE plots of (γ, β) of the ﬁrst (left) and last (right) FiLM layers of a 6-FiLM layer Network. FiLM parameters
cluster by low-level reasoning functions in the ﬁrst layer and by high-level reasoning functions in the last layer.

in a speciﬁc case. Together, these ﬁndings suggest that FiLM
learns to selectively upregulate, downregulate, and shut off
feature maps based on conditioning information.

FiLM Parameters t-SNE Plot
In Figure 6, we visualize
FiLM parameter vectors (γ, β) for 3,000 random valida-
tion points with t-SNE. We analyze the deeper, 6-ResBlock
version of our model, which has a similar validation accu-
racy as our 4-ResBlock model, to better examine how FiLM
layers in different layers of a hierarchy behave. First and
last layer FiLM (γ, β) are grouped by the low-level and
high-level reasoning functions necessary to answer CLEVR
questions, respectively. For example, FiLM parameters for
equal color and query color are close for the ﬁrst
layer but apart for the last layer. The same is true for shape,
size and material questions. Conversely, equal shape,
equal size, and equal material FiLM parameters
are grouped in the last layer but split in the ﬁrst layer — like-
wise for other high level groupings such as integer compar-
ison and querying. These ﬁndings suggest that FiLM layers
learn a sort of function-based modularity without an archi-
tectural prior. Simply with end-to-end training, FiLM learns
to handle not only different types of questions differently,
but also different types of question sub-parts differently; the
FiLM model works from low-level to high-level processes
as is the proper approach. For models with fewer FiLM lay-
ers, such patterns also appear, but less clearly; these models
must begin higher level reasoning sooner.

4.3 Ablations

Using the validation set, we conduct an ablation study on our
best model to understand how FiLM learns visual reasoning.
We show results for test time ablations in Figure 7, for archi-
tectural ablations in Table 2, and for varied model depths in
Table 3. Without hyperparameter tuning, most architectural
ablations and model depths outperform prior state-of-the-art
on training from only image-question-answer triplets, sup-
porting FiLM’s overall robustness. Table 3 also shows using
the validation set that our results are statistically signiﬁcant.

Figure 7: An analysis of how robust FiLM parameters are to
noise at test time. The horizontal lines correspond to setting
γ or β to their respective training set mean values.

Effect of γ and β To test the effect of γ and β separately,
we trained one model with a constant γ = 1 and another
with β = 0. With these models, we ﬁnd a 1.5% and .5%
accuracy drop, respectively; FiLM can learn to condition the
CNN for visual reasoning through either biasing or scaling
alone, albeit not as well as conditioning both together. This
result also suggests that γ is more important than β.

To further compare the importance of γ and β, we run
a series of test time ablations (Figure 7) on our best, fully-
trained model. First, we replace β with the mean β across
the training set. This ablation in effect removes all condition-
ing information from β parameters during test time, from a
model trained to use both γ and β. Here, we ﬁnd that ac-
curacy only drops by 1.0%, while the same procedure on γ
results in a 65.4% drop. This large difference suggests that,
in practice, FiLM largely conditions through γ rather than β.
Next, we analyze performance as we add increasingly more
Gaussian noise to the best model’s FiLM parameters at test
time. Noise in gamma hurts performance signiﬁcantly more,
showing FiLM’s higher sensitivity to changes in γ than in β
and corroborating the relatively greater importance of γ.

Restricting γ To understand what aspect of γ is most ef-
fective, we train a model that limits γ to (0, 1) using sig-

Overall

Model

Overall Model

Overall

Model

Restricted γ or β

FiLM with β := 0
FiLM with γ := 1
FiLM with γ := σ(γ)
FiLM with γ := tanh(γ)
FiLM with γ := exp(γ)

Moving FiLM within ResBlock

FiLM after residual connection
FiLM after ResBlock ReLU-2
FiLM after ResBlock Conv-2
FiLM before ResBlock Conv-1

Removing FiLM from ResBlocks

No FiLM in ResBlock 4
No FiLM in ResBlock 3-4
No FiLM in ResBlock 2-4
No FiLM in ResBlock 1-4

Miscellaneous

1 × 1 conv only, with no coord. maps
No residual connection
No batch normalization
Replace image features with raw pixels

96.9
95.9
95.9
96.3
96.3

96.6
97.7
97.1
95.0

96.8
96.5
97.3
21.4

95.3
94.0
93.7
97.6

Best Architecture

97.4±.4

Table 2: CLEVR val accuracy for ablations, trained with the
best architecture with only speciﬁed changes. We report the
standard deviation of the best model accuracy over 5 runs.

moid, as many models which use feature-wise, multiplica-
tive gating do. Likewise, we also limit γ to (−1, 1) using
tanh. Both restrictions hurt performance, roughly as much
as removing conditioning from γ entirely by training with
γ = 1. Thus, FiLM’s ability to scale features by large mag-
nitudes appears to contribute to its success. Limiting γ to
(0, ∞) with exp also hurts performance, validating the value
of FiLM’s capacity to negate and zero out feature maps.

Conditional Normalization We perform an ablation
study on the placement of FiLM to evaluate the relation-
ship between normalization and FiLM that Conditional Nor-
malization approaches assume. Unfortunately, it is difﬁcult
to accurately decouple the effect of FiLM from normaliza-
tion by simply training our corresponding model without
normalization, as normalization signiﬁcantly accelerates,
regularizes, and improves neural network learning (Ioffe
and Szegedy 2015), but we include these results for com-
pleteness. However, we ﬁnd no substantial performance
drop when moving FiLM layers to different parts of our
model’s ResBlocks; we even reach the upper end of the
best model’s performance range when placing FiLM after
the post-normalization ReLU in the ResBlocks. Thus, we
decouple the name from normalization for clarity regarding
where the fundamental effectiveness of the method comes
from. By demonstrating this conditioning mechanism is not
closely connected to normalization, we open the doors to ap-
plications other settings in which normalization is less com-
mon, such as RNNs and reinforcement learning, which are

1 ResBlock
2 ResBlocks
3 ResBlocks
4 ResBlocks
5 ResBlocks

93.5
97.1
96.7
97.4±.4
97.4

6 ResBlocks
7 ResBlocks
8 ResBlocks
12 ResBlocks

97.7
97.4
97.6
96.9

Table 3: CLEVR val accuracy by FiLM model depth.

promising directions for future work with FiLM.

Repetitive Conditioning To understand the contribution
of repetitive conditioning towards FiLM model success, we
train FiLM models with successively fewer FiLM layers.
Models with fewer FiLM layers, even a single FiLM layer,
do not deviate far from the best model’s performance, reveal-
ing that the model can reason and answer diverse questions
successfully by modulating features even just once. This ob-
servation highlights the capacity of even one FiLM layer.
Perhaps one FiLM layer can pass enough question informa-
tion to the CNN to enable it to carry out reasoning later in
the network, in place of the more hierarchical conditioning
deeper FiLM models appear to use. We leave more in-depth
investigation of this matter for future work.

Spatial Reasoning To examine how FiLM models ap-
proach spatial reasoning, we train a version of our best
model architecture, from image features, with only 1 × 1
convolutions and without feeding coordinate feature maps
indicating relative spatial position to the model. Due to the
global max-pooling near the end of the model, this model
cannot transfer information across spatial positions. No-
tably, this model still achieves a high 95.3% accuracy, in-
dicating that FiLM models are able to reason about space
simply from the spatial information contained in a single lo-
cation of ﬁxed image features.

Residual Connection Removing the residual connection
causes one of the larger accuracy drops. Since there is a
global max-pooling operation near the end of the network,
this ﬁnding suggests that the best model learns to primar-
ily use features of locations that are repeatedly important
throughout lower and higher levels of reasoning to make its
ﬁnal decision. The higher accuracies for models with FiLM
modulating features inside residual connections rather than
outside residual connections supports this hypothesis.

Model Depth Table 3 shows model performance by the
number of ResBlocks. FiLM is robust to varying depth but
less so with only 1 ResBlock, backing the earlier theory that
the FiLM-ed network reasons throughout its pipeline.

4.4 CLEVR-Humans: Human-Posed Questions
To assess how well visual reasoning models generalize
to more realistic, complex, and free-form questions, the
CLEVR-Humans dataset was introduced (Johnson et al.
2017b). This dataset contains human-posed questions on
CLEVR images along with their corresponding answers.
The number of samples is limited — 18K for training, 7K

Q: What object
color
of
Cylinder

is the
grass? A:

Q: Which shape objects
are partially obscured
from view? A: Sphere

Q: What color is the
matte object farthest to
the right? A: Brown

shape

Q: What
is
reﬂecting in the large
cube? A: Cylinder

If all cubical ob-
Q:
jects were removed what
shaped objects would
there be the most of? A:
Sphere (P: Rubber)

Figure 8: Examples from CLEVR-Humans, which introduces new words (underlined) and concepts. After ﬁne-tuning on
CLEVR-Humans, a CLEVR-trained model can now reason about obstruction, superlatives, and reﬂections but still struggles
with hypothetical scenarios (rightmost). It also has learned human preference to primarily identify objects by shape (leftmost).

for validation, and 7K for testing. The questions were col-
lected from Amazon Mechanical Turk workers prompted to
ask questions that were likely hard for a smart robot to an-
swer. As a result, CLEVR-Humans questions use more di-
verse vocabulary and complex concepts.

Method To test FiLM on CLEVR-Humans, we take our
best CLEVR-trained FiLM model and ﬁne-tune its FiLM-
generating linguistic pipeline alone on CLEVR-Humans.
Similar to prior work (Johnson et al. 2017b), we do not
update the visual pipeline on CLEVR-Humans to mitigate
overﬁtting to the small training set.

Results Our model achieves state-of-the-art generalization
to CLEVR-Humans, both before and after ﬁne-tuning, as
shown in Table 4, indicating that FiLM is well-suited to han-
dle more complex and diverse questions. Figure 8 shows ex-
amples from CLEVR-Humans with FiLM model answers.
Before ﬁne-tuning, FiLM outperforms prior methods by a
smaller margin. After ﬁne-tuning, FiLM reaches a consider-
ably improved ﬁnal accuracy. In particular, the gain in ac-
curacy made by FiLM upon ﬁne-tuning is more than 50%
greater than those made by other models; FiLM adapts data-
efﬁciently using the small CLEVR-Humans dataset.

the prior

Notably, FiLM surpasses

state-of-the-art
method, Program Generator + Execution Engine (PG+EE),
after ﬁne-tuning by 9.3%. Prior work on PG+EEs explains
that this neural module network method struggles on ques-
tions which cannot be well approximated with the model’s
module inventory (Johnson et al. 2017b). In contrast, FiLM
has the freedom to modulate existing feature maps, a fairly
ﬂexible and ﬁne-grained operation, in novel ways to reason
about new concepts. These results thus provide some evi-
dence for the beneﬁts of FiLM’s general nature.

4.5 CLEVR Compositional Generalization Test

To test how well models learn compositional concepts that
generalize, CLEVR-CoGenT was introduced (Johnson et
al. 2017a). This dataset is synthesized in the same way as
CLEVR but contains two conditions: in Condition A, all
cubes are gray, blue, brown, or yellow and all cylinders are

Model

Train
CLEVR

Train CLEVR,
ﬁne-tune human

LSTM
CNN+LSTM
CNN+LSTM+SA+MLP
PG+EE (18K prog.)

CNN+GRU+FiLM

27.5
37.7
50.4
54.0

56.6

36.5
43.2
57.6
66.6

75.9

Table 4: CLEVR-Humans test accuracy, before (left) and
after (right) ﬁne-tuning on CLEVR-Humans data

red, green, purple, or cyan; in Condition B, cubes and cylin-
ders swap color palettes. Both conditions contain spheres of
all colors. CLEVR-CoGenT thus indicates how a model an-
swers CLEVR questions: by memorizing combinations of
traits or by learning disentangled or general representations.

Results We train our best model architecture on Condition
A and report accuracies on Conditions A and B, before and
after ﬁne-tuning on B, in Figure 9. Our results indicate FiLM
surpasses other visual reasoning models at learning general
concepts. FiLM learns better compositional generalization
even than PG+EE, which explicitly models compositional-
ity and is trained with program-level supervision that specif-
ically includes ﬁltering colors and ﬁltering shapes.

Sample Efﬁciency and Catastrophic Forgetting We
show sample efﬁciency and forgetting curves in Figure 9.
FiLM achieves prior state-of-the-art accuracy with 1/3 as
much ﬁne-tuning data. However, our FiLM model still suf-
fers from catastrophic forgetting after ﬁne-tuning.

Zero-Shot Generalization FiLM’s accuracy on Condi-
tion A is much higher than on B, suggesting FiLM has mem-
orized attribute combinations to an extent. For example, the
model learns a bias that cubes are not cyan, as learning this
training set bias helps minimize training loss.

To overcome this bias, we develop a novel FiLM-based
zero-shot generalization method. Inspired by word embed-
ding manipulations, e.g. “King” - “Man” + “Woman” =
“Queen” (Mikolov et al. 2013), we test if linear manipula-

Method

Train A
B

A

Fine-tune B
A

B

CNN+LSTM+SA
PG+EE (18K prog.)
CNN+GRU+FiLM
CNN+GRU+FiLM 0-Shot

80.3
96.6
98.3
98.3

68.7
73.7
75.6
78.8

75.7
76.1
80.8
81.1

75.8
92.7
96.9
96.9

Figure 9: CoGenT results. FiLM ValB accuracy reported on
ValB without the 30K ﬁne-tuning samples (Figure). Accu-
racy before and after ﬁne-tuning on 30K of ValB (Table).

tion extends to reasoning with FiLM. We compute (γ, β)
for “How many cyan cubes are there?” via the linear com-
bination of questions in the FiLM parameter space: “How
many cyan spheres are there?” + “How many brown cubes
are there?” − “How many brown spheres are there?”. With
this (γ, β), our model can correctly count cyan cubes. We
show another example of this method in Figure 10.

We evaluate this method on validation B, using a parser to
automatically generate the right combination of questions.
We test previously reported CLEVR-CoGenT FiLM mod-
els with this method and show results in Figure 9. With this
method, there is a 3.2% overall accuracy gain when train-
ing on A and testing for zero-shot generalization on B. Yet
this method could only be applied to 1/3 of questions in
B. For these questions, model accuracy starts at 71.5% and
jumps to 80.7%. Before ﬁne-tuning on B, the accuracy be-
tween zero-shot and original approaches on A is identical,
likewise for B after ﬁne-tuning. We note that difference in
the predicted FiLM parameters between these two methods
is negligible, likely causing the similar performance.

We achieve these improvements without speciﬁcally
training our model for zero-shot generalization. Our method
simply allows FiLM to take advantage of any concept dis-
entanglement in the CNN after training. We also observe
that convex combinations of the FiLM parameters – i.e. be-
tween “How many cyan things are there?” and “How many
brown things are there?” – often monotonically interpolates
the predicted answer between the answers to endpoint ques-
tions. These results highlight, to a limited extent, the ﬂexi-
bility of FiLM parameters for meaningful manipulations.

As implemented, this method has many limitations. How-
ever, approaches from word embeddings, representation
learning, and zero-shot learning can be applied to directly
optimize (γ, β) for analogy-making (Bordes et al. 2013;
Guu, Miller, and Liang 2015; Oh et al. 2017). The FiLM-ed
network could directly train with this procedure via back-
propagation. A learned model could also replace the parser.
We ﬁnd such avenues promising for future work.

What is the blue big cylinder made of?
Question
What is the blue big sphere made of?
(1) Swap shape
What is the green big cylinder made of?
(2) Swap color
(3) Swap shape/color What is the green big sphere made of?

Figure 10: A CLEVR-CoGenT example. The combination
of concepts “blue” and “cylinder” is not in the training
set. Our zero-shot method computes the original question’s
FiLM parameters via linear combination of three other ques-
tions’ FiLM parameters: (1) + (2) - (3). This method corrects
our model’s answer from “rubber” to “metal”.

5 Conclusion
We show that a model can achieve strong visual reasoning
using general-purpose Feature-wise Linear Modulation lay-
ers. By efﬁciently manipulating a neural network’s interme-
diate features in a selective and meaningful manner using
FiLM layers, a RNN can effectively use language to mod-
ulate a CNN to carry out diverse and multi-step reasoning
tasks over an image. Our ablation study suggests that FiLM
is resilient to architectural modiﬁcations, test time ablations,
and even restrictions on FiLM layers themselves. Notably,
we provide evidence that FiLM’s success is not closely con-
nected with normalization as previously assumed. Thus, we
open the door for applications of this approach to settings
where normalization is less common, such as RNNs and re-
inforcement learning. Our ﬁndings also suggest that FiLM
models can generalize better, more sample efﬁciently, and
even zero-shot to foreign or more challenging data. Overall,
the results of our investigation of FiLM in the case of visual
reasoning complement broader literature that demonstrates
the success of FiLM-like techniques across many domains,
supporting the case for FiLM’s strength not simply within a
single domain but as a general, versatile approach.

6 Acknowledgements
We thank the developers of PyTorch (pytorch.org)
and (Johnson et al. 2017b) for open-source code which
our implementation was based off. We thank Mohammad
Pezeshki, Dzmitry Bahdanau, Yoshua Bengio, Nando de
Freitas, Hugo Larochelle, Laurens van der Maaten, Joseph
Cohen, Joelle Pineau, Olivier Pietquin, J´er´emie Mary, C´esar
Laurent, Chin-Wei Huang, Layla Asri, Max Smith, and
James Ough for helpful discussions and Justin Johnson
for CLEVR test evaluations. We thank NVIDIA for do-
nating a DGX-1 computer used in this work. We also ac-
knowledge FRQNT through the CHIST-ERA IGLU project,
Coll`ege Doctoral Lille Nord de France, and CPER Nord-
Pas de Calais/FEDER DATA Advanced data science and
technologies 2015-2020 for funding our work. Lastly, we
thank acronymcreator.net for the acronym FiLM.

References
Anderson, P.; He, X.; Buehler, C.; Teney, D.; Johnson, M.; Gould,
S.; and Zhang, L. 2017. Bottom-up and top-down attention for
image captioning and vqa. In VQA Workshop at CVPR.
Andreas, J.; Marcus, R.; Darrell, T.; and Klein, D. 2016a. Learning
to compose neural networks for question answering. In NAACL.
Andreas, J.; Rohrbach, M.; Darrell, T.; and Klein, D. 2016b. Neural
module networks. In CVPR.
Antol, S.; Agrawal, A.; Lu, J.; Mitchell, M.; Batra, D.; Zitnick,
C. L.; and Parikh, D. 2015. VQA: Visual Question Answering. In
ICCV.
Bordes, A.; Usunier, N.; Garcia-Duran, A.; Weston, J.; and
Yakhnenko, O. 2013. Translating embeddings for modeling multi-
relational data.
In Burges, C. J. C.; Bottou, L.; Welling, M.;
Ghahramani, Z.; and Weinberger, K. Q., eds., NIPS. Curran As-
sociates, Inc. 2787–2795.
Chung, J.; G¨ulc¸ehre, C¸ .; Cho, K.; and Bengio, Y. 2014. Empirical
evaluation of gated recurrent neural networks on sequence model-
ing. In Deep Learning Workshop at NIPS.
de Vries, H.; Strub, F.; Mary, J.; Larochelle, H.; Pietquin, O.; and
Courville, A. C. 2017. Modulating early visual processing by lan-
guage. In NIPS.
Dumoulin, V.; Shlens, J.; and Kudlur, M. 2017. A learned repre-
sentation for artistic style. In ICLR.
Eigen, D.; Ranzato, M.; and Sutskever, I. 2014. Learning factored
representations in a deep mixture of experts. In ICLR Workshops.
Gehring, J.; Auli, M.; Grangier, D.; Yarats, D.; and Dauphin, Y. N.
2017. Convolutional sequence to sequence learning. In ICML.
Geman, D.; Geman, S.; Hallonquist, N.; and Younes, L. 2015. Vi-
sual turing test for computer vision systems. volume 112, 3618–
3623. National Acad Sciences.
Ghiasi, G.; Lee, H.; Kudlur, M.; Dumoulin, V.; and Shlens, J. 2017.
Exploring the structure of a real-time, arbitrary neural artistic styl-
ization network. CoRR abs/1705.06830.
Goyal, Y.; Khot, T.; Summers-Stay, D.; Batra, D.; and Parikh, D.
2017. Making the V in VQA matter: Elevating the role of image
understanding in Visual Question Answering. In CVPR.
Guu, K.; Miller, J.; and Liang, P. 2015. Traversing knowledge
graphs in vector space. In EMNLP.
Ha, D.; Dai, A.; and Le, Q. 2016. Hypernetworks. In ICLR.
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learn-
ing for image recognition. In CVPR.
Hochreiter, S., and Schmidhuber, J. 1997. Long short-term mem-
ory. Neural Comput. 9(8):1735–1780.
Hu, R.; Andreas, J.; Rohrbach, M.; Darrell, T.; and Saenko, K.
2017. Learning to reason: End-to-end module networks for visual
question answering. In ICCV.
Hu, J.; Shen, L.; and Sun, G. 2017. Squeeze-and-Excitation Net-
works. In ILSVRC 2017 Workshop at CVPR.
Huang, X., and Belongie, S. 2017. Arbitrary style transfer in real-
time with adaptive instance normalization. In ICCV.
Ioffe, S., and Szegedy, C. 2015. Batch normalization: Accelerat-
ing deep network training by reducing internal covariate shift. In
ICML.
Johnson, J.; Hariharan, B.; van der Maaten, L.; Fei-Fei, L.; Zitnick,
C. L.; and Girshick, R. B. 2017a. CLEVR: A diagnostic dataset
for compositional language and elementary visual reasoning.
In
CVPR.

Johnson, J.; Hariharan, B.; van der Maaten, L.; Hoffman, J.; Li, F.;
Zitnick, C. L.; and Girshick, R. B. 2017b. Inferring and executing
programs for visual reasoning. In ICCV.
Jordan, M. I., and Jacobs, R. A. 1994. Hierarchical mixtures of
experts and the em algorithm. Neural Comput. 6(2):181–214.
Kim, T.; Song, I.; and Bengio, Y. 2017. Dynamic layer normaliza-
tion for adaptive neural acoustic modeling in speech recognition.
In InterSpeech.
Kingma, D. P., and Ba, J. 2015. Adam: A method for stochastic
optimization. In ICLR.
Kirkpatrick, J.; Pascanu, R.; Rabinowitz, N.; Veness, J.; Desjardins,
G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho, T.; Grabska-
Barwinska, A.; Hassabis, D.; Clopath, C.; Kumaran, D.; and Had-
sell, R. 2017. Overcoming catastrophic forgetting in neural net-
works. National Academy of Sciences 114(13):3521–3526.
Lu, J.; Yang, J.; Batra, D.; and Parikh, D.
2016. Hierarchi-
cal question-image co-attention for visual question answering. In
NIPS.
Malinowski, M., and Fritz, M. 2014. A multi-world approach
to question answering about real-world scenes based on uncertain
input. In NIPS.
Malinowski, M.; Rohrbach, M.; and Fritz, M. 2015. Ask your
neurons: A neural-based approach to answering questions about
images. In ICCV.
Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J.
2013. Distributed representations of words and phrases and their
compositionality. In NIPS.
Oh, J.; Singh, S.; Lee, H.; and Kholi, P. 2017. Zero-shot task gen-
eralization with multi-task deep reinforcement learning. In ICML.
Perez, E.; de Vries, H.; Strub, F.; Dumoulin, V.; and Courville,
A. C. 2017. Learning visual reasoning without strong priors. In
MLSLP Workshop at ICML.
Radford, A.; Metz, L.; and Chintala, S. 2016. Unsupervised rep-
resentation learning with deep convolutional generative adversarial
networks. In ICLR.
Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.;
Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M. S.; Berg, A. C.;
and Li, F. 2015. Imagenet large scale visual recognition challenge.
IJCV 115(3):211–252.
Santoro, A.; Raposo, D.; Barrett, D. G.; Malinowski, M.; Pascanu,
R.; Battaglia, P.; and Lillicrap, T. 2017. A simple neural network
module for relational reasoning. CoRR abs/1706.01427.
Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton,
G.; and Dean, J. 2017. Outrageously large neural networks: The
sparsely-gated mixture-of-experts layer. In ICLR.
van den Oord, A.; Dieleman, S.; Zen, H.; Simonyan, K.; Vinyals,
O.; Graves, A.; Kalchbrenner, N.; Senior, A.; and Kavukcuoglu,
K. 2016a. Wavenet: A generative model for raw audio. CoRR
abs/1609.03499.
van den Oord, A.; Kalchbrenner, N.; Espeholt, L.; Vinyals, O.;
Graves, A.; and Kavukcuoglu, K. 2016b. Conditional image gen-
eration with pixelcnn decoders. In NIPS.
van der Maaten, L., and Hinton, G. 2008. Visualizing data using
t-sne. JMLR 9(Nov):2579–2605.
Watters, N.; Tacchetti, A.; Weber, T.; Pascanu, R.; Battaglia, P.;
2017. Visual interaction networks. CoRR
and Zoran, D.
abs/1706.01433.
Yang, Z.; He, X.; Gao, J.; Deng, L.; and Smola, A. J. 2016. Stacked
attention networks for image question answering. In CVPR.

7 Appendix

7.1 Error Analysis

We examine the errors our model makes to understand where our
model fails and how it acts when it does. Examples of these errors
are shown in Figures 12 and 13.

Occlusion Many model errors are due to partial occlusion.
These errors may likely be ﬁxed using a CNN that operates at a
higher resolution, which is feasible since FiLM has a computa-
tional cost that is independent of resolution.

Counting 96.1% of counting mistakes are off-by-one errors,
showing FiLM has learned underlying concepts behind counting
such as close relationships between close numbers.

Logical Consistency The model sometimes makes curious
reasoning mistakes a human would not. For example, we ﬁnd a
case where our model correctly counts one gray object and two
cyan objects but simultaneously answers that there are the same
number of gray and cyan objects. In fact, it answers that the num-
ber of gray objects is both less than and equal to the number of
yellow blocks. These errors could be prevented by directly mini-
mizing logical inconsistency, an interesting avenue for future work
orthogonal to FiLM.

7.2 Model Details

Rather than output γi,c directly, we output ∆γi,c, where:

γi,c = 1 + ∆γi,c,

(3)

since initially zero-centered γi,c can zero out CNN feature map
activations and thus gradients. In our implementation, we opt to
output ∆γi,c rather than γi,c, but for simplicity, throughout our pa-
per, we explain FiLM using γi,c. However, this modiﬁcation does
not seem to affect our model’s performance on CLEVR statistically
signiﬁcantly.

We present training and validation curves for best model trained
from image features in Figure 11. We observe fast accuracy gains
initially, followed by slow, steady increases to a best validation ac-
curacy of 97.84%, at which point training accuracy is 99.53%.
We train on CLEVR for 80 epochs, which takes 4 days using 1
NVIDIA TITAN Xp GPU when learning from image features. For
practical reasons, we stop training on CLEVR after 80 epochs, but
we observe that accuracy continues to increase slowly even after-
wards.

Figure 11: Best model training and validation curves.

Q: Is there a big brown ob-
ject of the same shape as the
green thing? A: Yes (P: No)

Q: What number of other
things are the same material
as the big gray cylinder? A:
6 (P: 5)

Q: What shape is the big
metal thing that is the same
color as the small cylinder?
A: Cylinder (P: Sphere)

Q: How many other things
are the same material as the
tiny sphere? A: 3 (P: 2)

Figure 12: Some image-question pairs where our model pre-
dicts incorrectly. Most errors we observe are due to partially
occluded objects, as highlighted in the three ﬁrst examples.

Question
How many gray things are there?
How many cyan things are there?
Are there as many gray things as cyan things?
Are there more gray things than cyan things?
Are there fewer gray things than cyan things?

Answer
1
2
Yes
No
Yes

Figure 13: An interesting failure example where our model
counts correctly but compares counts erroneously. Its third
answer is incorrect and inconsistent with its other answers.

7.3 What Do FiLM Layers Learn?
We visualize FiLM’s effect on a single arbitrary feature map in Fig-
ures 14 and 15. We also show histograms of per-layer γi,c values,
per-layer βi,c values, and per-channel FiLM parameter statistics in
Figures 16, 17, and 18, respectively.

M
L
i
F
e
r
o
f
e
B

M
L
i
F
r
e
t
f

A

M
L
i
F
e
r
o
f
e
B

M
L
i
F
r
e
t
f

A

Feature 14 - Block 1

Feature 14 - Block 1

Q: What is the color of
the large rubber cylin-
der? A: Cyan

Q: What is the color
of
the large rubber
sphere? A: Gray

Q: What is the color of
the cube? A: Yellow

Q: How many cylin-
ders are there? A: 4

Q: What is the color of
the large rubber cylin-
der? A: Yellow

Q: What is the color
of
the large rubber
sphere? A: Gray

Q: What is the color of
the cube? A: Yellow

Q: How many cylin-
ders are there? A: 4

Figure 14: Visualizations of feature map activations (scaled from 0 to 1) before and after FiLM for a single arbitrary feature
map from the ﬁrst ResBlock. This particular feature map seems to detect gray and brown colors. Interestingly, FiLM modiﬁes
activations for speciﬁcally colored objects for color-speciﬁc questions but leaves activations alone for color-agnostic questions.
Note that since this is the ﬁrst FiLM layer, pre-FiLM activations (Rows 1 and 3) for all questions are identical, and differences
in post-FiLM activations (Rows 2 and 4) are solely due FiLM’s use of question information.

M
L
i
F
e
r
o
f
e
B

i

M
L
F
r
e
t
f

A

Feature 79 - Block 4

Q: How many cyan
objects are behind the
gray sphere? A: 2

Q: How many cyan
objects are in front of
the gray sphere? A: 1

Q: How many cyan
objects are left of the
gray sphere? A: 2

Q: How many cyan
objects are right of the
gray sphere? A: 1

Figure 15: Visualization of the impact of FiLM for a single arbitrary feature map from the last ResBlock. This particular feature
map seems to focus on spatial features (i.e. front/back or left/right) Note that since this is the last FiLM layer, the top row
activations have already been inﬂuenced by question information via several FiLM layers.

Figure 16: Histograms of γi,c values for each FiLM layer (layers 1-4 from left to right), computed on CLEVR’s validation set.
Plots are scaled identically. FiLM layers appear gradually more selective and higher variance.

Figure 17: Histograms of βi,c values for each FiLM layer (layers 1-4 from left to right) computed on CLEVR’s validation set.
Plots are scaled identically. βi,c values take a different, higher variance distribution in the ﬁrst layer than in later layers.

Figure 18: Histograms of per-channel γc and βc statistics (mean and standard deviation) computed on CLEVR’s validation set.
From left to right: γc means, γc standard deviations, βc means, βc standard deviations. Different feature maps are modulated by
FiLM in different patterns; some are often zero-ed out while other rarely are, some are consistently scaled or shifted by similar
values while others by high variance values, etc.


