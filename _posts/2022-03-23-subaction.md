---
layout: post
title: "Exploring Sub-Actions via Node Attention Module"
tags: [Linux,Python]

---
<ul id="toc"></ul>

---
<div id="" style="text-align: justify;" markdown="1">
This article introduces how I designed the *Node Attention Module* from scratch under certain constraints. 
Our goal is to provide a plug-and-play module that extracts sub-actions on top of any existing action detection approach.
</div>
---

<a id="fig1"></a>
{% include image.html
   img="/data/subaction/demo.gif"
   caption="Fig. 1: A complex human activity is usually sub-divided into unit-actions [[19][Hou_2017_BMVC]]."
%}



<div id="" style="text-align: center;" markdown="1">
__A one-size-fits-all HCI solution extracts generic sub-actions that are shared across datasets.__
</div>

<div id="" style="text-align: justify;" markdown="1">
Deep learning models are state-of-the-art for action recognition tasks, but they frequently treat complex activities as singular objectives
and lack interpretability. Therefore, recent works have started to tackle the problem of exploring sub-actions in complex activities. 
In this article, we introduce a novel approach that explores the temporal structure of detected action instances by explicitly modeling 
sub-actions and benefiting from them. To this end, we propose to learn sub-actions as latent concepts and to explore them via the *Node Attention Module (NAM)*. 
The proposed method maps both visual and temporal representations to a latent space where the sub-actions are learned discriminatively in an end-to-end fashion. 
The result is a set of latent vectors that can be interpreted as cluster centers in the embedding space. *NAM* is highly modular and extensible, 
and can easily be combined with existing deep learning models for various other video-related tasks in the future. 
</div>

---

### Introduction

<a id="fig2"></a>
{% include image.html
   img="/data/subaction/hussein2019videograph.png"
   caption="Fig. 2: The activity of “preparing coffee” can be represented as an undirected graph of unit-actions [[3][hussein2019videograph]]."
%}

<div id="" style="text-align: justify;" markdown="1">
In recent years, deep learning has dominated many computer vision tasks, especially action recognition — an important research field, 
since almost all real-world videos contain multiple actions, and each action is composed of several sub-actions 
[[3][hussein2019videograph], [17][Piergiovanni2017Subevents], [14][Piergiovanni_2018_CVPR]]. As shown in <a href="#fig2">Fig. 2</a>, for instance, the activity of "preparing coffee" can be represented as an undirected graph of unit-actions,
 including "take cup", "pour coffee", "pour sugar", and "stir coffee".

<a id="fig3"></a>
{% include image.html
   img="/data/subaction/Piergiovanni2017Subevents.png"
   caption="Fig. 3: In a video of a basketball game, shooting and blocking events must occur nearby [[17][Piergiovanni2017Subevents]]."
%}

Furthermore, detecting the frames of one activity in a video should benefit from information in the frames corresponding to another activity.
As shown in <a href="#fig3">Fig. 3</a>, a blocking event cannot occur without a shooting event.



<a id="fig4"></a>
{% include image.html
   img="/data/subaction/Huang2021Modeling.png"
   caption="Fig. 4: Different action instances can share similar motion patterns [[16][Huang2021Modeling]]."
%}
By the same token, a complex action is inherently a temporal composition of sub-actions, which means sub-actions may have contextual relations — in other words, sub-actions of the same action should appear together in the corresponding video.
As shown in <a href="#fig4">Fig. 4</a>, for example, the “jump” sub-action in the red box always appears in its intra-class instances and is also shared across several actions.

</div>

---

### Challenges

Action recognition is the problem of identifying events performed by humans given a video input. There are two primary challenges:
+ Many high-level activities are composed of multiple temporal parts with different durations and speeds.

	--> **Implicitly model sub-actions to preserve their essential properties**

+ Usually, only video- or frame-level category labels are given; the sub-actions are undefined and not annotated. 

	--> **Represent video features via a group of sub-actions, i.e., the sub-action family.**

---

### Problem Formulation

**We would like to perform frame-wise inference.** Existing approaches explore sub-actions from a given video frame 
[[17][Piergiovanni2017Subevents], [14][Piergiovanni_2018_CVPR]] or from given video segments [[15][Long_2019_CVPR], [12][swetha2021unsupervised]]. 
Among them, the approaches that take video segments as input rely on multiple timestamps to extract features, which means that information from several timestamps must be considered to capture context from adjacent frames. 


Learning to represent videos is important; it requires embedding the spatial and temporal information of a series of frames. 
A Convolutional Neural Network (CNN) followed by a Recurrent Neural Network (RNN) is a backbone widely used to extract spatiotemporal information. 
In the past few years, extracting spatiotemporal information with 3D convolutional networks, such as I3D [[18][Carreira_2017_CVPR]], has also received a great deal of attention. 
In this article, we adopt recurrent methods (i.e., a CNN followed by an RNN) instead of 3D convolutional networks, in consideration of the efficiency of real-time per-frame inference and the model size.

---

### Proposed Model

<div id="" style="text-align: justify;" markdown="1">
**Node Attention Learning**

<a id="fig5"></a>
{% include image.html
   img="/data/subaction/hussein2019videograph2.png"
   caption="Fig. 5: Illustration from [[3][hussein2019videograph]]."
%}

Inspired by [[3][hussein2019videograph]], in a dataset of human activities, unit-actions can be thought of as the dominant latent short-range concepts; 
that is, unit-actions are the building blocks of human activities. As shown in <a href="#fig5">Fig. 5</a>, in order to associate sub-actions across actions, we introduce a set of vectors as a memory bank of sub-action templates. 
These serve as our sub-action pool and are projected into a meaningful latent space; we call them latent concepts.

There are three key points worth mentioning:
1. The sub-action family contains multiple feature vectors, where each vector is responsible for representing a specific sub-action.
2. The sub-action family is automatically discovered and shared among all actions in the dataset, while all actions contribute to the learning of the sub-action family.
3. The number of sub-actions is automatically determined, and they are found to be semantically meaningful.

<a id="fig6"></a>
{% include image.html
   img="/data/subaction/hussein2019videograph3.png"
   caption="Fig. 6: Node Attention Module [[3][hussein2019videograph]]."
%}

Although the properties of this work meet our goal, there are some concerns. First of all, it cannot perform frame-wise inference.
Second, the model is hard to train, since it is asked to learn the relationships among the nodes by itself.
To tackle this, we made the following modifications: (1) the embedding network extracts per-frame deep features using a 2D CNN instead of 3D networks; 
(2) we borrow the way of learning the relationships among the nodes from [[12][swetha2021unsupervised]].


<div id="" style="text-align: justify;" markdown="1">
**Sub-Action Exploration Loss** 
<a id="fig7"></a>
{% include image.html
   img="/data/subaction/swetha2021unsupervised.png"
   caption="Fig. 7: Unsupervised sub-action learning in complex activities [[12][swetha2021unsupervised]]."
%}

As shown in <a href="#fig7">Fig. 7</a>, the objective of [[12][swetha2021unsupervised]] is to learn latent concepts that represent potential sub-actions. 
The similarity between the latent concept of a sub-action and its most confident input features is maximized, while the similarity with respect to other input features is minimized. 

<a id="fig8"></a>
{% include image.html
   img="/data/subaction/cmp.png"
   caption="Fig. 8: Disentangled latent concept learning."
%}

Different from [[12][swetha2021unsupervised]], we consider only the k-th self-similarity (<a href="#fig8">Fig. 8(b)</a>) instead of the summary of latent concepts (<a href="#fig8">Fig. 8(a)</a>).
</div>

<div id="" style="text-align: justify;" markdown="1">

**Additional Constraints** 
+ **Diversity Loss** 
In terms of diversity exploration, we design a diversity loss to encourage each projected sub-action template in the memory bank to be different from the other templates (i.e., to be unique). 
This loss is calculated as the mean of the pairwise similarities over all sub-actions. 
More specifically, the pairwise similarity between sub-actions is measured with the dot product, and the resulting similarity matrix is encouraged to approximate the identity matrix under the Frobenius norm. 
Doing so makes the sub-actions mutually independent. 
</div>

<div id="" style="text-align: justify;" markdown="1">
+ **Background Suppression Regularization** 
Since the *Sub-Action Exploration Loss* encourages only foreground segments to produce high logits for specific sub-actions, background segments are left unsupervised. 
As mentioned in [[13][Lee2020BackgroundMV]], the softmax scores of some background segments can still be high due to the relative nature of the softmax function. 
Moreover, while the diversity loss encourages the latent concepts in the memory bank to be unique, it does not guarantee that each latent concept is meaningful. 
For instance, a latent concept may not represent any sub-action and may have low similarities with all input features during training. 
To tackle this, we follow [[13][Lee2020BackgroundMV]] and force background segments to have a uniform probability distribution over sub-actions. 
Doing so prevents background segments from having a high score for any sub-action.
</div>

<div id="" style="text-align: justify;" markdown="1">
+ **Length Regularization** 
The idea behind *NAM* is that **"an action must consist of at least two sub-actions."** 
To this end, we introduce a proportional compression regularization term with an action summary ratio, similar to 
[[10][ping2021exploring]], which penalizes overly long explored sub-actions and prevents the trivial solution.
</div>

<div id="" style="text-align: justify;" markdown="1">
+ **Foreground Entropy Regularization** 
We further introduce a standard entropy regularization term for foreground segments, which directly encourages less entropic
(more peaked) distributions. Contrary to the *Background Suppression Regularization*, the goal of entropy regularization is to alter the
attention maps by biasing them toward low entropy.
</div>

---

### Why Plug and Play?
One might think that [[3][hussein2019videograph]] has already addressed both the action and sub-action detection tasks from the perspective of graph theory, and that we merely propose an alternative way to improve its performance — 
so why do we still need plug and play? Indeed, [[3][hussein2019videograph]] is a multi-task learning method, and the dependency on sub-action family representations significantly degrades its action detection performance.
That is why we position our approach as *plug and play*. 

In our work, given a trained action detection model, we simply freeze its weights and apply *NAM* to extract sub-actions. 
This means we tackle the problem in two stages: in the first stage, an embedding based on visual and temporal information is learned; in the second stage, clustering is applied in this embedding space.

---

### Interface Design
We provide a plug-and-play Python module based on the PyTorch framework. An example of using the module is as follows.

{% highlight Python %}
def NodeAttentionModule(x):
    # x: feature tensor, output of RNN backbone
    # x’s size: (B, T, D)
    # nodes: node tensor, must be initialized at beginning
    # nodes’s size: (D, N)
    
    # Q path
    x = relu(bn(x))
    x = fc_x(x)

    # K path
    n = relu(bn2(nodes))
    n = fc_n(n)

    # Q and K path
    A = matmul(x, n)
    A = softmax(A, dim=-1)
    
    return A
{% endhighlight %}

---

### Results

<a id="fig9"></a>
{% include image.html
   img="/data/subaction/output.png"
   caption="Fig. 9: Visualization of our sub-action results."
%}

---

### Conclusions and Future Work
First, our framework uses prototypes to represent sub-actions, which are learned automatically in an end-to-end manner. 
Second, the sub-action mechanism learns directly from individual videos and does not require triplet samples, which avoids a complicated sampling process. 
Besides, during learning, all videos interact with the same sub-action family, which provides a holistic solution for studying all available videos. 
Moreover, the proposed method operates at the sub-action level and bridges videos from different categories. 
We proposed the plug-and-play *Node Attention Module (NAM)* to predict the action units within an action in videos. 
To our knowledge, this is the first work to explore a plug-and-play module for sub-action representation learning, capturing the temporal structure and relationships within an action. 

---

### References

[[1][kaidi2019fewshot]]
Cao, Kaidi, et al. Few-shot video classification via temporal alignment.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.


[[2][rahman2016optimizing]] Md Atiqur Rahman and Yang Wang. Optimizing
intersection-over-union in deep neural networks for image
segmentation. In Advances in Visual Computing, pages
234–244, Cham, 2016. Springer International Publishing.


[[3][hussein2019videograph]] Noureldien Hussein, Efstratios Gavves, and Arnold W. M.
Smeulders. Videograph: Recognizing minutes-long human
activities in videos, 2019.

[[4][li2017concurrent]] Xinyu Li, Yanyi Zhang, Jianyu Zhang, Shuhong Chen,
Ivan Marsic, Richard A. Farneth, and Randall S. Burd.
Concurrent activity recognition with multimodal cnn-lstm
structure, 2017.

[[5][Donahue_2015_CVPR]] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadar-
rama, Marcus Rohrbach, Subhashini Venugopalan, Kate
Saenko, and Trevor Darrell. Long-term recurrent convolu-
tional networks for visual recognition and description. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), June 2015.


[[6][Sigurdsson_2017_CVPR]] Gunnar A. Sigurdsson, Santosh Divvala, Ali Farhadi, and
Abhinav Gupta. Asynchronous temporal fields for action
recognition. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), July 2017.

[[7][Xu_2017_ICCV]] Huijuan Xu, Abir Das, and Kate Saenko. R-c3d: Region
convolutional 3d network for temporal activity detection.
In Proceedings of the IEEE International Conference on
Computer Vision (ICCV), Oct 2017.


[[8][Tran_2018_CVPR]] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann
LeCun, and Manohar Paluri. A closer look at spatiotem-
poral convolutions for action recognition. In Proceedings
of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), June 2018.


[[9][zhenzhi2020boundary]] Limin Wang Zhifeng Li Zhenzhi Wang, Ziteng Gao and
Gangshan Wu. Boundary-aware cascade networks for
temporal action segmentation. In Computer Vision – ECCV
2020, pages 34–51, Cham, 2020. Springer International
Publishing. ISBN 978-3-030-58595-2.

[[10][ping2021exploring]] Ping Li, Qinghao Ye, Luming Zhang, Li Yuan, Xianghua
Xu, and Ling Shao. Exploring global diverse atten-
tion via pairwise temporal relation for video summariza-
tion. Pattern Recognition, 111:107677, 2021. ISSN 0031-3203. doi: https://doi.org/10.1016/j.patcog.2020. 107677. URL https://www.sciencedirect.com/
science/article/pii/S0031320320304805.

[[11][Luo_2021_CVPR]] Wang Luo, Tianzhu Zhang, Wenfei Yang, Jingen Liu, Tao
Mei, Feng Wu, and Yongdong Zhang. Action unit memory
network for weakly supervised temporal action localiza-
tion. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
9969–9979, June 2021.

[[12][swetha2021unsupervised]] Sirnam Swetha, Hilde Kuehne, Yogesh S Rawat, and
Mubarak Shah. Unsupervised discriminative embedding
for sub-action learning in complex activities, 2021.

[[13][Lee2020BackgroundMV]] Pilhyeon Lee, Jinglu Wang, Yan Lu, and H. Byun. Back-
ground modeling via uncertainty estimation for weakly-
supervised action localization. ArXiv, abs/2006.07006, 2020.


[[14][Piergiovanni_2018_CVPR]] AJ Piergiovanni and Michael S. Ryoo. Learning latent
super-events to detect multiple activities in videos. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), June 2018.


[[15][Long_2019_CVPR]] Fuchen Long, Ting Yao, Zhaofan Qiu, Xinmei Tian, Jiebo
Luo, and Tao Mei. Gaussian temporal awareness networks
for action localization. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), June 2019.

[[16][Huang2021Modeling]] Linjiang Huang, Yan Huang, Wanli Ouyang, and Liang
Wang. Modeling sub-actions for weakly supervised tem-
poral action localization. IEEE Transactions on Image
Processing, 30:5154–5167, 2021. doi: 10.1109/TIP.2021.
3078324.

[[17][Piergiovanni2017Subevents]] A. J. Piergiovanni, Chenyou Fan, and Michael S. Ryoo.
Learning latent subevents in activity videos using tem-
poral attention filters. In Proceedings of the Thirty-First
AAAI Conference on Artificial Intelligence, AAAI’17, page
4247–4254. AAAI Press, 2017.

[[18][Carreira_2017_CVPR]] Joao Carreira and Andrew Zisserman. Quo vadis, action
recognition? a new model and the kinetics dataset. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), July 2017.

[[19][Hou_2017_BMVC]] Hou, Rui, Rahul Sukthankar, and Mubarak Shah. 
Real-Time Temporal Action Localization in Untrimmed Videos by Sub-Action Discovery.
BMVC. Vol. 2. 2017.

---

[github]: https://github.com/Lilyo

[kaidi2019fewshot]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf
[rahman2016optimizing]:http://cs.umanitoba.ca/~ywang/papers/isvc16.pdf
[hussein2019videograph]: https://arxiv.org/abs/1905.05143
[li2017concurrent]: https://arxiv.org/abs/1702.01638
[Donahue_2015_CVPR]: https://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf
[Sigurdsson_2017_CVPR]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Sigurdsson_Asynchronous_Temporal_Fields_CVPR_2017_paper.pdf
[Xu_2017_ICCV]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Xu_R-C3D_Region_Convolutional_ICCV_2017_paper.pdf
[Tran_2018_CVPR]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf
[zhenzhi2020boundary]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700035.pdf
[ping2021exploring]: https://www.sciencedirect.com/science/article/abs/pii/S0031320320304805
[Luo_2021_CVPR]: https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Action_Unit_Memory_Network_for_Weakly_Supervised_Temporal_Action_Localization_CVPR_2021_paper.pdf
[swetha2021unsupervised]: https://arxiv.org/abs/2105.00067
[Lee2020BackgroundMV]: https://arxiv.org/pdf/2006.07006.pdf
[Piergiovanni_2018_CVPR]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Piergiovanni_Learning_Latent_Super-Events_CVPR_2018_paper.pdf
[Long_2019_CVPR]: http://staff.ustc.edu.cn/~xinmei/publications_pdf/2019/Gaussian%20Temporal%20Awareness%20Networks%20for%20Action%20Localization.pdf
[Huang2021Modeling]: https://ieeexplore.ieee.org/document/9430747
[Piergiovanni2017Subevents]: https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/15005/14307
[Sener_2018_CVPR]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sener_Unsupervised_Learning_and_CVPR_2018_paper.pdf
[Carreira_2017_CVPR]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf
[Hou_2017_BMVC]: http://www.bmva.org/bmvc/2017/papers/paper091/paper091.pdf