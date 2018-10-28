# 使用图像到图像转换的无限制面部几何重建

> Unrestricted Facial Geometry Reconstruction Using Image-to-Image Translation 

$$
\text{Matan Sela Elad Richardson Ron Kimmel}
\\
\text{Department of Computer Science, Technion - Israel Institute of Technology}
$$

$$
\text{{matansel,eladrich,ron}@cs.technion.ac.il}
$$

> 图1：提出的方法的结果。重建的几何图形显示在相应的输入图像旁边。

## 摘要

*最近已经表明，神经网络可以从单个给定图像恢复面部的几何结构。大多数现有面部几何重建方法的共同点是将解空间限制到一些低维子空间。虽然这样的模型显着地简化了重建问题，但它的表现力本质上是有限的。作为替代方案，我们提出了一种图像到图像转换网络，其将输入图像联合映射到深度图像和面部对应图。然后，可以利用这种基于像素的显式映射，使用纯粹的几何改进过程，在极端表情下提供各种面部的高质量重建。在最近的方法的精神中，网络仅用合成数据训练，然后在“野外”面部图像上进行评估。定性和定量分析都证明了我们方法的准确性和稳健性。*

## 1. 简介

恢复面部的几何结构是计算机视觉中的基本任务，具有许多应用。例如，逼真电影中演员的面部特征可以通过精心设计的面部装备来手动编辑，以操纵表达[42]。在制作动画电影时，跨多个帧跟踪演员的几何形状允许将表达式转换为动画化身[14,1,8]。基于图像的人脸识别方法使恢复的几何形状变形，以在给定图像中产生输入面的中性和正面版本，减少同一主体的图像之间的变化[49,19]。至于医疗应用，获得面部结构允许精细规划美学操作和整形手术，设计个性化面具[2,37]甚至生物印刷面部器官。

在这里，我们专注于在广泛的表情和姿势下从单个面部图像恢复面部的几何结构。这个问题已经研究了几十年，并且大多数现有解决方案涉及以下组件中的一个或多个。

* 面部标志[25,46,32,47]  - 一组自动检测到的关键点，如鼻尖和眼角，可以指导重建过程[49,26,1， 12,29]。
* 参考面部模型 - 平均中性面，用作光学流程的初始化或来自着色程序的形状[19,26]。
* 一个三维可变形模型 - 一个合理的面部几何形状的先前低维线性子空间，它允许一个有效但粗糙的面部结构恢复[4,6,49,36,23,33,43]，

> 图2：算法重建管道。

虽然使用这些组件可以简化重建问题，但它们会引入一些固有的局限性。仅依赖于地标的方法仅限于稀疏的约束点集。使用参考面部模型的经典技术可能无法恢复极端表情和非正面姿势，因为光流限制变形到图像平面。可变形模型在提供一些鲁棒性的同时限制了重建，因为它只能表达粗糙的几何形状。将这些组件中的一些组合在一起可以缓解这些问题，然而，潜在的限制仍然体现在最终的重建中。

或者，我们提出一种不受限制的方法，其涉及完全卷积网络，其学习将输入面部图像转换为包含两个地图的表示。第一个地图是深度图像的估计，而第二个是在图像域中嵌入面部模板网格。该网络遵循[22]的图像到图像转换框架进行训练，其中引入了额外的基于法线的损耗以增强深度结果。与之前的方法类似，我们使用合成图像进行训练，其中图像是从各种面部身份，姿势，表情，光照条件，背景和材料参数中采样的。令人惊讶的是，尽管网络仍然使用有限生成模型绘制的面部进行训练，但它可以推广和生成远远超出该模型有限范围的结构。为了处理原始网络结果，使用迭代面部变形过程，其将表示组合成完整的面部网格。最后，应用改进步骤来产生详细的重建。神经网络与纯几何技术的这种新颖融合使我们能够仅从单个图像在中尺度上重建具有皱纹和细节的高质量网格。

虽然在过去[33,34,43,48,24]中提出了使用神经网络进行面部重建，但先前的方法仍然受到线性模型的表现力的限制。在[34]中，提出了第二个网络来重新进行粗面部重建，然而，它无法补偿超出给定子空间的大的几何变化。例如，鼻子的结构仍然受到面部可变形模型的跨度的限制。通过直接在图像域中学习无约束几何，我们克服了这一限制，正如定量和定性实验结果所证明的那样。为了进一步分析所提出的表示的潜力，我们设计了一个将图像从一个域转换到另一个域的应用程序。作为案例研究，我们将合成面部图像转换为现实图像，强制我们的网络作为损失函数，以在整个跨域映射中保留几何。

本文的主要贡献是：

* 一种新颖的公式，用于从单个图像预测面部的几何表示，不限于线性模型。
* 纯粹的几何变形和改进程序，利用网络表示产生高质量的面部重建。
* 所提出的网络的新颖应用，其允许将合成的面部图像转换为现实的图像，同时保持几何结构的完整性。

## 2. 概述

算法管道如图2所示。网络的输入是面部图像，网络产生两个输出：第一个是与输入图像对齐的估计深度图。第二输出是从每个像素到参考面部网格上的对应顶点的密集映射。为了使结果成为完整的顶点对应和完整的遮挡部分，我们通过迭代的非刚性变形过程在三维空间中扭曲模板网格。最后，由输入图像引导的细节重建算法恢复了面部的微妙几何结构。评估代码可在https://github.com/matansel/pix2vertex上找到

## 3.学习几何表示

在使用神经网络时，需要考虑几种设计选择。首先是训练数据，包括输入通道，标签以及如何收集样本。其次是架构的选择。一种常见的方法是从现有架构[27,39,40,20]开始，并使其适应手头的问题。最后，可以选择培训过程，包括损失标准和优化技术。接下来，我们将描述我们对这些元素的选择。

### 3.1 数据及其表示

建议的网络的目的是从给定的面部图像回归几何表示。该表示由以下两个组成部分组成：

**深度图像面部几何的深度剖面**。实际上，对于许多只提供深度剖面的面部重建任务是足够的[18,26]。

**对应图**。一种嵌入，允许将图像像素映射到模板面部模型上的点，以三角网格形式给出。要为任何面部几何体计算此签名，我们使用规范化规范面上相应点的x，y和z坐标绘制每个顶点。然后，我们使用相应投影顶点的颜色值绘制地图中的每个像素，请参见图3。此特征图是一种变形不可知表示，可用于面部动作捕捉[44]，面部规范化[49]和纹理映射[50]等应用。虽然在[34,48]中使用类似的表示作为迭代网络的反馈信道，但面部恢复仍然局限于面部可变形模型的跨度。

> 图3：参考模板面与不同视点的密集对应签名一起呈现。

> 图4：训练数据样本及其表示。

为了训练网络，我们采用[33]中提出的相同的合成数据生成程序。通过从面部变形模型[4]绘制随机网格坐标S和纹理T来生成每个随机面部。在实践中，我们绘制一对高斯随机向量αg和αt，并如下恢复合成面
$$
S=\mu_g+A_g\alpha_g\\T=\mu_t+A_t\alpha_t.
$$
其中μg和μt分别是模型的堆叠平均面部几何和纹理。 Ag和At是矩阵，其列是低维线性子空间的基础，分别跨越合理的面部几何和纹理。请注意，几何基础Ag由身份和表达基础元素组成，如[10]中所提出的。接下来，我们在各种照明条件和姿势下渲染随机纹理网格，生成合成面部图像的数据集。由于每个合成图像都知道地面实况几何，因此可以容易地将匹配深度和对应图用作标签。输入图像的一些示例以及它们所需的输出如图4所示。

使用合成数据在推广到“野外”图像时仍然存在一些空白[9,33]，但它提供了急需的灵活性。生成过程并确保从图像到其标签的确定性连接。或者，其他方法[16,43]提出通过采用现有的重建算法并将其结果视为地面实况来生成训练数据标签。例如，G¨uler等。 [16]，使用类似于[48]的框架，将密集的对应图匹配到面部图像的数据集，从一组稀疏的标志开始。然后将这些对应图用作其方法的训练标签。请注意，此类数据也可用于培训我们的网络，而无需任何其他修改。

### 3.2 图像到几何转换

像素预测需要适当的网络架构[30,17]。所提出的结构受到[22]中提出的最近的图像到图像转换框架的启发，其中训练网络以将输入图像映射到各种类型的输出图像。在那里使用的架构基于U-net [35]布局，其中在编码器和解码器中的相应层之间使用跳过连接。关于网络实施的其他考虑因素在补充中给出。

虽然在[22]中使用了L1和对抗性损失函数的组合，但在提出的框架中，我们选择省略对抗性损失。这是因为与[22]中探讨的问题不同，我们的设置包括较少的映射模糊性。因此，分布式损失函数效率较低，主要是引入伪像。尽管如此，由于基本的L1损失函数有利于深度预测中的稀疏误差并且没有考虑像素邻域之间的差异，因此它不足以产生精细的几何结构，参见图5b。因此，我们建议用额外的L1项来增加损失函数，这会惩罚重建深度和地面实况的法线之间的差异。
$$
L_N(\hat z,z)=\|\vec n(\hat z)-\vec n(z)\|_1,\tag1
$$
其中z是恢复的深度，z表示地面深度图像。在训练期间，我们设置λL1\x3d 100和λN\x3d 10，其中λL1和λN是匹配损失权重。注意，对于对应图像，仅应用L1损失。图5展示了LN对网络提供的深度重建质量的贡献。

> 图5：（a）输入图像，（b）仅具有L1损失函数的结果和（c）具有附加法线损失函数的结果。注意（b）中的工件。

## 4. 从表示到网格

基于得到的深度和对应关系，我们介绍了将2.5D表示转换为3D面部网格的方法。该过程由迭代弹性变形算法（4.1）组成，然后是由输入图像（4.2）驱动的细节恢复步骤。得到的输出是精确的重建面部网格，其具有与具有固定三角剖分的模板网格的完整顶点对应。这种类型的数据有助于各种动态面部处理应用，例如面部装备，其允许创建和编辑演员的照片般逼真的动画。作为副产品，该过程还通过完成面部中被错误地分类为背景的一部分的域来校正网络的预测。

### 4.1. 非刚性注册

接下来，我们描述基于迭代变形的注册管道。首先，我们通过连接相邻像素将深度图从网络转换为网格。基于来自网络的对应图，我们计算从模板面到网格的自然变换。通过最小化相应顶点对之间的平方欧几里德距离来完成该操作。接下来，类似于[28]，迭代非刚性配准过程使变换后的模板变形，使其与网格对齐。请注意，在整个注册过程中，只有模板会变形，而目标网格仍保持固定状态。每次迭代都涉及以下四个步骤。

1. 通过评估对应嵌入空间中的最近邻居，模板网格中的每个顶点vi∈V与目标网格上的顶点ci相关联。该步骤与[28]中描述的方法不同，后者计算欧几里德空间中的最近邻居。结果，所提出的步骤允许使用任意表达将单个模板面注册到不同的面部身份。
2. 在下一步中检测并忽略物理上远离的对（vi，ci）和正常方向不一致的对。
3. 通过最小化以下能量使模板网格变形

$$
\begin{align}
E(V,C)\quad=\quad&\alpha_{p2point}\sum_{(v_i,c_i)\in\mathcal J}\|v_i-c_i\|_2^2
\\&+\alpha_{p2plane}\sum_{(v_i,c_i)\in\mathcal J}|\vec n(c_i)(v_i-c_i)|^2
\\&+\alpha_{memb}\sum_{i\in\mathcal V}\sum_{v_j\in\mathcal N(v_i)}w_{i,j}\|v_i-v_j\|_2^2,\tag2
\end{align}
$$

其中，wi，j是对应于双调和拉普拉斯算子的权重（见[21,5]），（cid：126）n（ci）是目标网格ci处的对应顶点的法线，J是剩余关联顶点对（vi，ci）的集合，并且N（vi）是关于顶点vi的集合1环相邻顶点。请注意，上面的第一个术语是匹配之间的欧几里德距离的平方和。第二项是在目标网格的对应点处从点vi到切平面的距离。第三项量化网格的刚度。

4. 如果模板网格在当前迭代和前一个迭代之间的运动低于固定阈值，我们将权重αmemb除以2。这放松了刚度项，并允许在下一次迭代中产生更大的变形。

当刚度重量低于给定阈值时，该迭代过程终止。补充材料中提供了注册过程的进一步实施信息和参数。此阶段的结果输出是具有固定三角测量的变形模板，其包含由网络恢复的整体面部结构，但是更平滑和完整，参见图9的第三列。

### 4.2. 精细细节重建

Although the network already recovers some ﬁne geometric details, such as wrinkles and moles, across parts of the face, a geometric approach can reconstruct details at a ﬁner level, on the entire face, independently of the resolution. Here, we propose an approach motivated by the passive-stereo facial reconstruction method suggested in [3]. The underlying assumption here is that subtle geometric structures can be explained by local variations in the image domain. For some skin tissues, such as nevi, this assumption is inaccurate as the intensity variation results from the albedo. In such cases, the geometric structure would be wrongly modiﬁed. Still, for most parts of the face, the reconstructed details are consistent with the actual variations in depth.     The method begins from an interpolated version of the deformed template. Each vertex v ∈ VD is painted with the intensity value of the nearest pixel in the image plane. Since we are interested in recovering small details, only the high spatial frequencies, µ(v), of the texture, τ (v), are taken into consideration in this phase. For computing this frequency band, we subtract the synthesized low frequencies from the original intensity values. This low-pass ﬁltered part can be computed by convolving the texture with a spatially varying Gaussian kernel in the image domain, as originally proposed. In contrast, since this convolution is equivalent to computing the heat distribution upon the shape after time dt, where the initial heat proﬁle is the original texture, we 
虽然网络已经在面部的各个部分恢复了一些细微的几何细节，例如皱纹和痣，但是几何方法可以在整个面上的细节水平上重建细节，而与分辨率无关。在这里，我们提出了一种由[3]中提出的被动 - 立体面部重建方法推动的方法。这里的基本假设是细微的几何结构可以通过图像域中的局部变化来解释。对于一些皮肤组织，例如痣，这种假设是不准确的，因为强度变化是由反照率引起的。在这种情况下，几何结构将被错误地修改。尽管如此，对于脸部的大多数部分，重建的细节与深度的实际变化一致。该方法从变形模板的插值版本开始。每个顶点v∈VD用图像平面中最近像素的强度值绘制。由于我们对恢复小细节感兴趣，因此在该阶段仅考虑纹理的高空间频率μ（v）τ（v）。为了计算该频带，我们从原始强度值中减去合成的低频。如最初提出的，可以通过将纹理与图像域中的空间变化高斯核卷积来计算该低通滤波部分。相比之下，由于这个卷积相当于计算在时间dt之后的形状上的热量分布，其中初始热量分布是原始纹理，我们

Figure 6: Mesoscopic displacement. From left to right: an input image, the shape after the iterative registration, the high-frequency part of the texture - µ(v), and the ﬁnal shape. 
图6：介观位移。从左到右：输入图像，迭代配准后的形状，纹理的高频部分 - μ（v）和最终形状。

propose to compute µ(v) as 
建议将μ（v）计算为

µ(v) = τ (v) − (I − dt · ∆g )−1 τ (v), 
μ（v）\x3dτ（v） - （I  -  dt·Δg）-1τ（v），

(3) where I is the identity matrix, ∆g is the cotangent weight discrete Laplacian operator for triangulated meshes [31], and dt is a scalar proportional to the cut-off frequency of the ﬁlter. Next, we displace each vertex along its normal direction such that v (cid:48) = v + δ(v)(cid:126)n(v). The step size of the displacement, δ(v), is a combination of a data-driven term, δµ (v), and a regularization one, δs (v). The data-driven term is guided by the high-pass ﬁltered part of the texture, µ(v). In practice, we require the local differences in the geometry to be proportional to the local variation in the high frequency band of the texture. For each vertex v , with a normal (cid:126)n(v), and a neighboring vertex vi , the data-driven term is given by 
（3）其中I是单位矩阵，Δg是三角网格的余切权重离散拉普拉斯算子[31]，而dt是与滤波器的截止频率成比例的标量。接下来，我们沿着其法线方向移动每个顶点，使得v（cid：48）\x3d v +δ（v）（cid：126）n（v）。位移的步长δ（v）是数据驱动项δμ（v）和正则化项δs（v）的组合。数据驱动的术语由纹理的高通滤波部分μ（v）引导。在实践中，我们要求几何中的局部差异与纹理的高频带中的局部变化成比例。对于每个顶点v，使用法线（cid：126）n（v）和相邻顶点vi，数据驱动项由下式给出：

(cid:16) 
（CID：16）

(cid:17) 
（CID：17）

(cid:80) 
（CID：80）

vi∈N (v) 
vi∈N（v）

δµ (v) = 
δμ（v）\x3d

α(v ,vi ) (µ(v) − µ(vi )) 
α（v，vi）（μ（v） - μ（vi））

1 − |(cid:104)v−vi ,(cid:126)n(v)(cid:105)| 
1  -  |（cid：104）v-vi，（cid：126）n（v）（cid：105）|

(cid:107)v−vi (cid:107) 
（cid：107）v-vi（cid：107）

α(v ,vi ) 
α（v，vi）

, 
，

(cid:80) 
（CID：80）

vi∈N (v) 
vi∈N（v）

(4) where α(v ,vi ) = exp (−(cid:107)v − vi (cid:107)). For further explanation of Equation 4, we refer the reader to the supplementary material of this paper or the implementation details of [3]. Since we move each vertex along the normal direction, triangles could intersect each other, particularly in domains of high curvature. To reduce the probability of such collisions, a regularizing displacement ﬁeld, δs (v), is added. This term is proportional to the mean curvature of the original surface, and is equivalent to a single explicit mesh fairing step [11]. The ﬁnal surface modiﬁcation is given by 
（4）其中α（v，vi）\x3d exp（ - （cid：107）v  -  vi（cid：107））。为了进一步解释公式4，我们请读者参考本文的补充材料或[3]的实现细节。由于我们沿法线方向移动每个顶点，因此三角形可以相互交叉，特别是在高曲率的域中。为了降低这种碰撞的可能性，增加了正则化位移场δs（v）。该项与原始曲面的平均曲率成比例，相当于单个显式网格整流步骤[11]。最终的表面修饰由下式给出

v (cid:48) = v + (ηδµ (v) + (1 − η)δs (v)) · (cid:126)n(v), 
v（cid：48）\x3d v +（ηδμ（v）+（1-η）δs（v））·（cid：126）n（v），

(5) for some constant η ∈ [0, 1]. A demonstration of the results before and after this step is presented in Figure 6 
（5）对于某些常数η∈[0,1]。图6显示了该步骤之前和之后的结果

5. Experiments 
  5.实验

Next, we present evaluations on both the proposed network and the pipeline as a whole, and comparison to different prominent methods of single image based facial reconstruction [26, 49, 34]. 
接下来，我们提出了对所提出的网络和管道整体的评估，并与基于单个图像的面部重建的不同突出方法进行比较[26,49,34]。

5.2. Quantitative Evaluation 
5.2。定量评估

For a quantitative comparison, we used the ﬁrst 200 subjects from the BU-3DFE dataset [45], which contains facial images aligned with ground truth depth images. Each method provides its own estimation for the depth image alongside a binary mask, representing the valid pixels to be taken into account in the evaluation. Obviously, since the problem of reconstructing depth from a single image is ill-posed, the estimation needs to be judged up to global scaling and transition along the depth axis. Thus, we compute these paramters using the Random Sample Concensus (RANSAC) approach [13], for normalizing the estimation according to the ground truth depth. This signiﬁcantly reduces the absolute error of each method as the global parameter estimation is robust to outliers. Note that the parameters of the RANSAC were identical for all the methods and samples. The results of this comparison are given in Table 1, where the units are given in terms of the percentile of the ground-truth depth range. As a further analysis of the reconstruction accuracy, we computed the mean absolute error of each method based on expressions, see Table 2. 
为了进行定量比较，我们使用了BU-3DFE数据集[45]中的前200个主题，其中包含与地面实况深度图像对齐的面部图像。每个方法提供其自身对深度图像的估计以及二元掩模，表示在评估中要考虑的有效像素。显然，由于从单个图像重建深度的问题是不适定的，因此需要判断估计直到全局缩放并沿着深度轴过渡。因此，我们使用随机样本一致性（RANSAC）方法[13]计算这些参数，以根据地面实况深度对估计进行归一化。这显着降低了每种方法的绝对误差，因为全局参数估计对异常值是稳健的。请注意，RANSAC的参数对于所有方法和样品都是相同的。该比较的结果在表1中给出，其中单位以地面实况深度范围的百分位数给出。作为对重建精度的进一步分析，我们基于表达式计算每种方法的平均绝对误差，参见表2。

Figure 9: The reconstruction stages. From left to right: the input image, the reconstruction of the network, the registered template and the ﬁnal shape. 
图9：重建阶段。从左到右：输入图像，网络重建，注册模板和最终形状。

Figure 7: Network Output. 
图7：网络输出。

Figure 8: Texture mapping via the embedding. 
图8：通过嵌入的纹理映射。

5.1. Qualitative Evaluation 
5.1。定性评估

The ﬁrst component of our algorithm is an Image-toImage network. In Figure 7, we show samples of output maps produced by the proposed network. Although the network was trained with synthetic data, with simple random backgrounds (see Figure 4), it successfully separates the hair and background from the face itself and learns the corresponding representations. To qualitatively assess the accuracy of the correspondence, we present a visualization where an average facial texture is mapped to the image plane via the predicted embedding, see Figure 8, this shows how the network successfully learns to represent the facial structure. Next, in Figure 9 we show the reconstruction of the network, alongside the registered template and the ﬁnal shape. Notice how the structural information retrieved by the network is preserved through the geometric stages. Figure 10 shows a qualitative comparison between the proposed method and others. One can see that our method better matches the global structure, as well as the facial details. To better perceive these differences, see Figure 11. Finally, to demonstrate the limited expressiveness of the 3DMM space compared to our method, Figure 12 presents our registered template next to its projection onto the 3DMM space. This clearly shows that our network is able to learn structures which are not spanned by the 3DMM model. 
我们算法的第一个组成部分是Image-toImage网络。在图7中，我们显示了由所提出的网络产生的输出映射的样本。虽然网络是用合成数据训练的，具有简单的随机背景（见图4），但它成功地将头发和背景与面部本身分开，并学习相应的表示。为了定性地评估对应的准确性，我们提出了一种可视化，其中平均面部纹理通过预测的嵌入被映射到图像平面，参见图8，这显示了网络如何成功地学习来表示面部结构。接下来，在图9中，我们显示了网络的重建，以及注册模板和最终形状。请注意网络检索的结构信息如何通过几何阶段保留。图10显示了所提出的方法与其他方法之间的定性比较。可以看出，我们的方法更好地匹配全局结构以及面部细节。为了更好地理解这些差异，请参见图11.最后，为了演示3DMM空间与我们的方法相比有限的表现力，图12显示了我们在3DMM空间投影旁边的注册模板。这清楚地表明我们的网络能够学习不被3DMM模型跨越的结构。

Input 
输入

Proposed 
建议

[34] 
[34]

[26] 
[26]

[49] 
[49]

Proposed 
建议

[34] 
[34]

[26] 
[26]

[49] 
[49]

Figure 10: Qualitative comparison. Input images are presented alongside the reconstructions of the different methods. 
图10：定性比较。输入图像与不同方法的重建一起呈现。

Mean Err. 3.89 3.85 3.61 
意思是错误。 3.89 3.85 3.61

3.51 
3.51

[26] [49] [34] Ours 
[26] [49] [34]我们的

Std Err. Median Err. 4.14 2.94 3.23 2.93 2.99 2.72 
Std Err。中位数错误。 4.14 2.94 3.23 2.93 2.99 2.72

2.69 
2.69

2.65 
2.65

90% Err. 7.34 7.91 6.82 
90％呃。 7.34 7.91 6.82

6.59 
6.59

Input 
输入

Proposed 
建议

[34] 
[34]

[26] 
[26]

[49] 
[49]

Figure 11: Zoomed qualitative result of ﬁrst and fourth subjects from Figure 10. 
图11：来自图10的第一和第四受试者的缩放定性结果。

5.3. The Network as a Geometric Constraint 
5.3。网络作为几何约束

As demonstrated by the results, the proposed network successfully learns both the depth and the embedding representations for a variety of images. This representation is the key part behind the reconstruction pipeline. However, it can also be helpful for other face-related tasks. As an example, we show that the network can be used as a geometric constraint for facial image manipulations, such as transforming synthetic images into realistic ones. This idea 
如结果所示，所提出的网络成功地学习了各种图像的深度和嵌入表示。这种表示是重建管道背后的关键部分。但是，它对其他与面部相关的任务也很有帮助。作为一个例子，我们表明网络可以用作面部图像处理的几何约束，例如将合成图像转换为现实图像。这个想法

Table 1: Quantitative evaluation on the BU-3DFE Dataset. From left to right: the absolute depth errors evaluated by mean, standard deviation, median and the average ninety percent largest error. 
表1：BU-3DFE数据集的定量评估。从左到右：绝对深度误差通过平均值，标准偏差，中位数和平均百分之九十的最大误差来评估。

AN 3.47 4.00 
AN 3.47 4.00

3.42 
3.42

3.67 
3.67

DI 4.03 3.93 3.46 
DI 4.03 3.93 3.46

3.34 
3.34

FE 3.94 3.91 3.64 
FE 3.94 3.91 3.64

3.36 
3.36

HA 4.30 3.70 3.41 
HA 4.30 3.70 3.41

3.01 
3.01

NE 3.43 3.76 4.22 
NE 3.43 3.76 4.22

3.17 
3.17

SA 3.52 3.61 3.59 
SA 3.52 3.61 3.59

3.37 
3.37

SU 4.19 
SU 4.19

3.96 
3.96

4.00 4.41 
4.00 4.41

[26] [49] [34] Ours 
[26] [49] [34]我们的

Table 2: The mean error by expression. From left to right: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise. 
表2：表达式的平均误差。从左到右：愤怒，厌恶，恐惧，快乐，中立，悲伤，惊喜。

is based on recent advances in applying Generative Adversarial Networks (GAN) [15] for domain adaption tasks [41]. learns to map from the source domain, DS , to the target doIn the basic GAN framework, a Generator Network (G) main DT , where a Discriminator Network (D) tries to dis
基于最近在域适应任务中应用生成对抗网络（GAN）[15]的进展[41]。学会从源域DS映射到目标doIn在基本GAN框架中，生成器网络（G）主DT，其中鉴别器网络（D）试图dis

Figure 12: 3DMM Projection. From left to right: the input image, the registered template, the projected mesh and the projection error. 
图12：3DMM投影。从左到右：输入图像，注册模板，投影网格和投影误差。

tinguish between generated images and samples from the target domain, by optimizing the following objective 
通过优化以下目标，在生成的图像和来自目标域的样本之间进行区分

min 
分

G 
G

V (D , G) = Ey∼DT [log D (y)] max + Ex∼DS [log (1 − D (G (x)))] . 
V（D，G）\x3d Ey〜DT [log D（y）] max + Ex_DS [log（1-D（G（x）））]。

D 
d

(6) 
（6）

Theoretically, this framework could also translate images from the synthetic domain into the realistic one. However, it does not guarantee that the underlying geometry of the synthetic data is preserved throughout that transformation. That is, the generated image might look realistic, but have a completely different facial structure from the synthetic input. To solve that potential inconsistency, we suggest to involve the proposed network as an additional loss function on the output of the generator. 
从理论上讲，该框架还可以将合成域中的图像转换为现实图像。但是，它并不能保证在整个转换过程中保留合成数据的基础几何。也就是说，生成的图像可能看起来很逼真，但是与合成输入具有完全不同的面部结构。为了解决这种潜在的不一致性，我们建议将所提出的网络作为发电机输出的附加损耗函数。

LGeom (x) = (cid:107)N et (x) − N et (G (x))(cid:107)1 , 
LGeom（x）\x3d（cid：107）N et（x） -  N et（G（x））（cid：107）1，

(7) where N et(·) represents the operation of the introduced network. Note that this is feasible, thanks to the fact that the proposed network is fully differentiable. The additional geometric ﬁdelity term forces the generator to learn a mapping that makes a synthetic image more realistic while keeping the underlying geometry intact. This translation process could potentially be useful for data generation procedures, similarly to [38]. Some successful translations are visualized in Figure 13. Notice that the network implicitly learns to add facial hair and teeth, and modify the texture the and shading, without changing the facial structure. As demonstrated by this analysis, the proposed network learns a strong representation that has merit not only for reconstruction, but for other tasks as well. 
（7）其中N et（·）表示引入网络的操作。请注意，这是可行的，这要归功于所提出的网络是完全可区分的。附加的几何形状术语迫使生成器学习一种映射，使合成图像更加真实，同时保持底层几何体的完整性。与[38]类似，这种翻译过程可能对数据生成过程有用。一些成功的翻译在图13中可视化。请注意，网络隐式学习添加面部毛发和牙齿，并修改纹理和阴影，而不改变面部结构。正如此分析所证明的那样，所提出的网络学习的强大代表性不仅适用于重建，也适用于其他任务。

6. Limitations 
  6.限制

One of the core ideas of this work was a model-free approach, where the solution space is not restricted by a low dimensional subspace. Instead, the Image-to-Image 
这项工作的核心思想之一是无模型方法，其中解空间不受低维子空间的限制。相反，图像到图像

Figure 13: Translation results. From top to bottom: synthetic input images, the correspondence and the depth maps recovered by the network, and the transformed result. 
图13：翻译结果。从上到下：合成输入图像，网络恢复的对应关系和深度图，以及转换结果。

network represents the solution in the extremely highdimensional image domain. This structure is learned from synthetic examples, and shown to successfully generalize to “in-the-wild” images. Still, facial images that signiﬁcantly deviate from our training domain are challenging, resulting in missing areas and errors inside the representation maps. More speciﬁcally, our network has difﬁculty handling extreme occlusions such as sunglasses, hands or beards, as these were not seen in the training data. Similarly to other methods, reconstructions under strong rotations are also not well handled. Reconstructions under such scenarios are shown in the supplementary material. Another limiting factor of our pipeline is speed. While the suggested network by itself can be applied efﬁciently, our template registration step is currently not optimized for speed and can take a few minutes to converge. 
网络代表极高维图像域中的解决方案。该结构是从合成示例中学习的，并且示出为成功地概括为“野外”图像。尽管如此，显着偏离我们训练领域的面部图像仍具有挑战性，导致表现图中缺少区域和错误。更具体地说，我们的网络难以处理极端遮挡，如太阳镜，手或胡须，因为这些在训练数据中没有出现。与其他方法类似，强旋转下的重建也没有得到很好的处理。在这种情况下的重建显示在补充材料中。我们管道的另一个限制因素是速度。尽管建议的网络本身可以有效地应用，但我们的模板注册步骤目前尚未针对速度进行优化，并且可能需要几分钟才能收敛。

7. Conclusion 
  7.结论

We presented an unrestricted approach for recovering the geometric structure of a face from a single image. Our algorithm employs an Image-to-Image network which maps the input image to a pixel-based geometric representation, followed by geometric deformation and reﬁnement steps. The network is trained only by synthetic facial images, yet, is capable of reconstructing real faces. Using the network as a loss function, we propose a framework for translating synthetic facial images into realistic ones while preserving the geometric structure. 
我们提出了一种不受限制的方法，用于从单个图像中恢复面部的几何结构。我们的算法采用图像到图像网络，将输入图像映射到基于像素的几何表示，然后是几何变形和重建步骤。网络仅由合成面部图像训练，但是能够重建真实面部。使用网络作为损失函数，我们提出了一个框架，用于将合成的面部图像转换为现实的图像，同时保留几何结构。

Acknowledgments 
致谢

We would like to thank Roy Or-El for the helpful discussions and comments. 
我们要感谢Roy Or-El的有益讨论和评论。

References 
参考

[1] O. Aldrian and W. A. Smith. A linear approach of 3D face shape and texture recovery using a 3d morphable model. In Proceedings of the British Machine Vision Conference, pages, pages 75–1, 2010. [2] I. Amirav, A. S. Luder, A. Halamish, D. Raviv, R. Kimmel, D. Waisman, and M. T. Newhouse. Design of aerosol face masks for children using computerized 3d face analysis. Journal of aerosol medicine and pulmonary drug delivery, 27(4):272–278, 2014. [3] T. Beeler, B. Bickel, P. Beardsley, B. Sumner, and M. Gross. High-quality single-shot capture of facial geometry. In ACM SIGGRAPH 2010 Papers, SIGGRAPH ’10, pages 40:1– 40:9, New York, NY, USA, 2010. ACM. [4] V. Blanz and T. Vetter. A morphable model for the synthesis of 3D faces. In Proceedings of the 26th annual conference on Computer graphics and interactive techniques, pages 187– 194. ACM Press/Addison-Wesley Publishing Co., 1999. [5] M. Botsch and O. Sorkine. On linear variational surface deformation methods. IEEE Transactions on Visualization and Computer Graphics, 14(1):213–230, Jan 2008. [6] P. Breuer, K.-I. Kim, W. Kienzle, B. Scholkopf, and V. Blanz. Automatic 3D face reconstruction from single images or video. In Automatic Face & Gesture Recognition, 2008. FG’08. 8th IEEE International Conference on, pages 1–8. IEEE, 2008. [7] C. Cao, D. Bradley, K. Zhou, and T. Beeler. Real-time highﬁdelity facial performance capture. ACM Transactions on Graphics (TOG), 34(4):46, 2015. [8] C. Cao, Y. Weng, S. Lin, and K. Zhou. 3D shape regression for real-time facial animation. ACM Transactions on Graphics (TOG), 32(4):41, 2013. [9] W. Chen, H. Wang, Y. Li, H. Su, D. Lischinsk, D. Cohen-Or, B. Chen, et al. Synthesizing training images for boosting human 3D pose estimation. arXiv preprint arXiv:1604.02703, 2016. [10] B. Chu, S. Romdhani, and L. Chen. 3d-aided face recognition robust to expression and pose variations. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 1907–1914. IEEE, 2014. [11] M. Desbrun, M. Meyer, P. Schr ¨oder, and A. H. Barr. Implicit fairing of irregular meshes using diffusion and curvature ﬂow. In Proceedings of the 26th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’99, pages 317–324, New York, NY, USA, 1999. ACM Press/Addison-Wesley Publishing Co. [12] P. Dou, Y. Wu, S. K. Shah, and I. A. Kakadiaris. Robust 3D face shape reconstruction from single images via two-fold coupled structure learning. In Proc. British Machine Vision Conference, pages 1–13, 2014. [13] M. A. Fischler and R. C. Bolles. Random sample consensus: a paradigm for model ﬁtting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. [14] P. Garrido, M. Zollh ¨ofer, D. Casas, L. Valgaerts, K. Varanasi, P. P ´erez, and C. Theobalt. Reconstruction of personalized 
[1] O. Aldrian和W. A. Smith。使用3d可变形模型的3D面部形状和纹理恢复的线性方法。参见英国机器视觉会议论文集，第75-1页，2010年。[2] I. Amirav，A。S. Luder，A。Halamish，D。Raviv，R。Kimmel，D。Waisman和M. T. Newhouse。用计算机三维人脸分析设计儿童气溶胶面罩。气溶胶医学和肺部给药杂志，27（4）：272-278,2014。[3] T. Beeler，B。Bickel，P。Beardsley，B。Sumner和M. Gross。高质量的单面捕捉面部几何体。在ACM SIGGRAPH 2010年论文中，SIGGRAPH \x26#39

3D face rigs from monocular video. ACM Transactions on Graphics (TOG), 35(3):28, 2016. [15] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014. [16] R. A. G ¨uler, G. Trigeorgis, E. Antonakos, P. Snape, S. Zafeiriou, and I. Kokkinos. Densereg: Fully convolutional dense shape regression in-the-wild. arXiv preprint arXiv:1612.01202, 2016. [17] B. Hariharan, P. Arbel ´aez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and ﬁne-grained localization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 447–456, 2015. [18] T. Hassner. Viewing real-world faces in 3d. In Proceedings of the IEEE International Conference on Computer Vision, pages 3607–3614, 2013. [19] T. Hassner, S. Harel, E. Paz, and R. Enbar. Effective face frontalization in unconstrained images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4295–4304, 2015. [20] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. [21] B. T. Helenbrook. Mesh deformation using the biharmonic operator. International journal for numerical methods in engineering, 56(7):1007–1021, 2003. [22] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Imageto-image translation with conditional adversarial networks. arXiv preprint arXiv:1611.07004, 2016. [23] L. Jiang, J. Zhang, B. Deng, H. Li, and L. Liu. 3d face reconstruction with geometry details from a single image. arXiv preprint arXiv:1702.05619, 2017. [24] A. Jourabloo and X. Liu. Large-pose face alignment via cnn-based dense 3D model ﬁtting. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016. [25] V. Kazemi and J. Sullivan. One millisecond face alignment with an ensemble of regression trees. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1867–1874, 2014. [26] I. Kemelmacher-Shlizerman and R. Basri. 3D face reconstruction from a single image using a single reference face shape. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(2):394–405, 2011. [27] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classiﬁcation with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. [28] H. Li. Animation Reconstruction of Deformable Surfaces. PhD thesis, ETH Zurich, November 2010. [29] F. Liu, D. Zeng, J. Li, and Q. Zhao. Cascaded regressor based 3D face reconstruction from a single arbitrary view image. arXiv preprint arXiv:1509.06161, 2015. [30] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3431–3440, 2015. 
来自单眼视频的3D面部装备。 ACM图形交易（TOG），35（3）：28,2016。[15] I. Goodfellow，J。Pouget-Abadie，M。Mirza，B。Xu，D。Warde-Farley，S。Ozair，A。 Courville和Y. Bengio。生成对抗网。 “神经信息处理系统的进展”，第2672-2680页，2014年。[16] R. A.G¨uler，G。Trigeorgis，E。Antonakos，P。Snape，S。Zafeiriou和I. Kokkinos。 Densereg：完全卷积密集的形状回归在野外。 arXiv preprint arXiv：1612.01202,2016。[17] B. Hariharan，P。Arbel\x26#39

[46] Z. Zhang, P. Luo, C. C. Loy, and X. Tang. Facial landmark detection by deep multi-task learning. In European Conference on Computer Vision, pages 94–108. Springer, 2014. [47] E. Zhou, H. Fan, Z. Cao, Y. Jiang, and Q. Yin. Extensive facial landmark localization with coarse-to-ﬁne convolutional network cascade. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 386–391, 2013. [48] X. Zhu, Z. Lei, X. Liu, H. Shi, and S. Z. Li. Face alignment across large poses: A 3d solution. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 146–155, 2016. [49] X. Zhu, Z. Lei, J. Yan, D. Yi, and S. Z. Li. High-ﬁdelity pose and expression normalization for face recognition in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 787–796, 2015. [50] G. Zigelman, R. Kimmel, and N. Kiryati. Texture mapping using surface ﬂattening via multidimensional scaling. IEEE Transactions on Visualization and Computer Graphics, 8(2):198–207, 2002. 
[46] Z. Zhang，P。Luo，C。C. Loy和X. Tang。通过深度多任务学习进行面部地标检测。在欧洲计算机视觉会议上，第94-108页。 Springer，2014。[47] E. Zhou，H。Fan，Z。Cao，Y。Jiang和Q. Yin。广泛的面部标志性定位，具有粗到细的卷积网络级联。在IEEE国际计算机视觉研讨会会议论文集，第386-391页，2013年。[48] X. Zhu，Z。Lei，X。Liu，H。Shi和S. Z. Li。大型姿势的脸部对齐：3D解决方案。在IEEE计算机视觉和模式识别会议论文集，第146-155页，2016年。[49] X. Zhu，Z。Lei，J。Yan，D。Yi和S. Z. Li。野外人脸识别的高保真姿态和表情归一化。在IEEE计算机视觉和模式识别会议论文集，第787-796页，2015年。[50] G. Zigelman，R。Kimmel和N. Kiryati。使用表面fl通过多维缩放进行纹理映射。 IEEE Transactions on Visualization and Computer Graphics，8（2）：198-207,2002。

[31] M. Meyer, M. Desbrun, P. Schr ¨oder, A. H. Barr, et al. Discrete differential-geometry operators for triangulated 2manifolds. Visualization and mathematics, 3(2):52–58, 2002. [32] S. Ren, X. Cao, Y. Wei, and J. Sun. Face alignment at 3000 fps via regressing local binary features. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1685–1692, 2014. [33] E. Richardson, M. Sela, and R. Kimmel. 3D face reconstruction by learning from synthetic data. In 3D Vision (3DV), 2016 International Conference on, pages 460–469. IEEE, 2016. [34] E. Richardson, M. Sela, R. Or-El, and R. Kimmel. Learning detailed face reconstruction from a single image. arXiv preprint arXiv:1611.05053, 2016. [35] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015. [36] S. Saito, L. Wei, L. Hu, K. Nagano, and H. Li. Photorealistic facial texture inference using deep neural networks. arXiv preprint arXiv:1612.00523, 2016. [37] M. Sela, N. Toledo, Y. Honen, and R. Kimmel. Customized facial constant positive air pressure (cpap) masks. arXiv preprint arXiv:1609.07049, 2016. [38] A. Shrivastava, T. Pﬁster, O. Tuzel, J. Susskind, W. Wang, and R. Webb. Learning from simulated and unsupervised images through adversarial training. arXiv preprint arXiv:1612.07828, 2016. [39] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. [40] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015. [41] Y. Taigman, A. Polyak, and L. Wolf. Unsupervised crossdomain image generation. arXiv preprint arXiv:1611.02200, 2016. [42] J. Thies, M. Zollhofer, M. Stamminger, C. Theobalt, and M. Nießner. Face2face: Real-time face capture and reenactment of rgb videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2387– 2395, 2016. [43] A. T. Tran, T. Hassner, I. Masi, and G. Medioni. Regressing robust and discriminative 3d morphable models with a very deep neural network. arXiv preprint arXiv:1612.04904, 2016. [44] T. Weise, S. Bouaziz, H. Li, and M. Pauly. Realtime performance-based facial animation. In ACM Transactions on Graphics (TOG), volume 30, page 77. ACM, 2011. [45] L. Yin, X. Wei, Y. Sun, J. Wang, and M. J. Rosato. A 3d facial expression database for facial behavior research. In Automatic face and gesture recognition, 2006. FGR 2006. 7th international conference on, pages 211–216. IEEE, 2006. 
[31] M. Meyer，M。Desbrun，P。Schr¨oder，A。H. Barr，et al。三角形2流形的离散微分几何算子。可视化与数学，3（2）：52-58,2002。[32] S. Ren，X。Cao，Y。Wei和J. Sun.通过回归本地二进制功能，以3000 fps进行面对齐。在IEEE计算机视觉和模式识别会议论文集，第1685-1692页，2014年。[33] E. Richardson，M。Sela和R. Kimmel。通过学习合成数据进行3D人脸重建。在3D Vision（3DV），2016年国际会议上，第460-469页。 IEEE，2016。[34] E. Richardson，M。Sela，R。Or-El和R. Kimmel。从单个图像学习详细的面部重建。 arXiv preprint arXiv：1611.05053,2016。[35] O. Ronneberger，P。Fischer和T. Brox。 U-net：用于生物医学图像分割的卷积网络。在医学图像计算和计算机辅助干预国际会议上，第234-241页。 Springer，2015。[36] S. Saito，L。Wei，L。Hu，K。Nagano和H. Li。基于深度神经网络的逼真面部纹理推理。 arXiv preprint arXiv：1612.00523,2016。[37] M. Sela，N。Toledo，Y。Honen和R. Kimmel。定制面部恒定正气压（cpap）面罩。 arXiv preprint arXiv：1609.07049,2016。[38] A. Shrivastava，T。P fi ster，O。Tuzel，J。Susskind，W。Wang和R. Webb。通过对抗训练学习模拟和无监督图像。 arXiv preprint arXiv：1612.07828,2016。[39] K. Simonyan和A. Zisserman。用于大规模图像识别的非常深的卷积网络。 arXiv preprint arXiv：1409.1556,2014。[40] C. Szegedy，W。Liu，Y。Jia，P。Sermanet，S。Reed，D。Anguelov，D。Erhan，V。Vanhoucke和A. Rabinovich。进一步深化卷积。在IEEE计算机视觉和模式识别会议论文集，第1-9页，2015年。[41] Y. Taigman，A。Polyak和L. Wolf。无监督的跨域图像生成。 arXiv preprint arXiv：1611.02200,2016。[42] J. Thies，M。Zollhofer，M。Stamminger，C。Theobalt和M.Nießner。 Face2face：rgb视频的实时面部捕捉和重演。在IEEE计算机视觉和模式识别会议论文集，第2387-2395页，2016年。[43] A.T.Tran，T。Hassner，I。Masi和G. Medioni。使用非常深的神经网络回归稳健且具有辨别力的3d可变形模型。 arXiv preprint arXiv：1612.04904,2016。[44] T. Weise，S。Bouaziz，H。Li和M. Pauly。基于实时性能的面部动画。在ACM Transactions on Graphics（TOG），第30卷，第77页.ACM，2011。[45] L. Yin，X。Wei，Y。Sun，J。Wang和M. J. Rosato。面部行为研究的3d面部表情数据库。自动脸部和手势识别，2006.FGR 2006.第7届国际会议，第211-216页。 IEEE，2006。

Supplementary Material 
补充材料

A. Additional Network Details 
A.其他网络详细信息

Here, we summarize additional considerations concerning the network and its training procedure. • The proposed architecture is based on the one introduced in [22]. For allowing further reﬁnement of the results, three additional convolution layers with a kernel of size 1 × 1 were concatenated at the end. Following the notations of [22], the encoder architecture is given as 
在这里，我们总结了有关网络及其培训程序的其他注意事项。 •建议的架构基于[22]中介绍的架构。为了允许进一步改进结果，最后将三个具有1×1大小的内核的卷积层连接起来。遵循[22]的符号，编码器架构如下

C 64 − C 128 − C 256 − C 512 − C 512 − C 512 − C 512 − C 512, 
C 64  -  C 128  -  C 256  -  C 512  -  C 512  -  C 512  -  C 512  -  C 512，

while the decoder is given by 
而解码器是由

CD512 − CD512 − CD512 − C 512 − C 512 − C 256 − C 128 − C 64 − C ∗ 64 − C ∗ 32 − C ∗ 4, 
CD512  -  CD512  -  CD512  -  C 512  -  C 512  -  C 256  -  C 128  -  C 64  -  C * 64  -  C * 32  -  C * 4，

where C ∗ represents a 1 × 1 convolution with stride 1. • The resolution of the input and output training images was 512 × 512 pixels. While this is a relatively large input size for training, the Image-to-Image architecture was able to process it successfully, and provided accurate results. Although, one could train a network on smaller resolutions and then evaluate it on larger images, as shown in [22], we found that our network did not successfully scale up for unseen resolutions. • While a single network was successfully trained to retrieve both depth and correspondence representations, our experiments show that training separated networks to recover the representations is preferable. Note that the architectures of both networks were identical. This can be justiﬁed by the observation that during training, a network allocates its resources for a speciﬁc translation task and the representation maps we used have different characteristics. • A necessary parameter for the registration step is the scale of the face with respect to the image dimensions. While this can be estimated based on global features, such as the distance between the eyes, we opted to retrieve it directly by training the network to predict the x and y coordinates of each pixel in the image alongside the z coordinate. 
其中C *表示步幅1的1×1卷积。•输入和输出训练图像的分辨率为512×512像素。虽然这是一个相对较大的训练输入大小，但Image-to-Image架构能够成功处理它，并提供准确的结果。虽然，人们可以在较小的分辨率上训练网络，然后在较大的图像上进行评估，如[22]所示，我们发现我们的网络没有成功扩展到看不见的分辨率。 •虽然成功训练了单个网络以检索深度和对应表示，但我们的实验表明，训练分离的网络以恢复表示是更可取的。请注意，两个网络的架构都是相同的。这可以通过以下观察得到证实：在训练期间，网络为特定的翻译任务分配其资源，并且我们使用的表示图具有不同的特征。 •注册步骤的必要参数是面部相对于图像尺寸的比例。虽然这可以基于全局特征（例如眼睛之间的距离）来估计，但我们选择通过训练网络直接检索它以预测图像中每个像素的x和y坐标以及z坐标。

B. Additional Registration and Reﬁnement Details 
B.附加注册和补充细节

Next, we provide a detailed version of the iterative deformation-based registration phase, including implementation details of the ﬁne detail reconstruction. 
接下来，我们提供基于迭代变形的注册阶段的详细版本，包括细节重建的实现细节。

B.1. Non-Rigid Registration 
B.1。非刚性注册

First, we turn the x,y and z maps from the network into a mesh, by connecting four neighboring pixels, for which the coordinates are known, with a couple of triangles. This step yields a target mesh that might have holes but has dense map to our template model. Based on the correspondence given by the network, we compute the afﬁne transformation from a template face to the mesh. This operation is done by minimizing the squared Euclidean distances between corresponding vertex pairs. To handle outliers, a RANSAC approach is used [13] with 1, 000 iterations and a threshold of 3 millimeters for detecting inliers. Next, similar to [28], an iterative non-rigid registration process deforms the transformed template, aligning it with the mesh. Note, that throughout the registration, only the template is warped, while the target mesh remains ﬁxed. Each iteration involves the following four steps. 
首先，我们将x，y和z地图从网络转换为网格，通过连接四个相邻的像素，其中坐标是已知的，具有几个三角形。此步骤生成一个目标网格，该网格可能具有孔但具有到我们的模板模型的密集地图。基于网络给出的对应关系，我们计算从模板面到网格的自然变换。通过最小化相应顶点对之间的平方欧几里德距离来完成该操作。为了处理异常值，使用RANSAC方法[13]，其中1000次迭代和3毫米的阈值用于检测内点。接下来，类似于[28]，迭代非刚性配准过程使变换后的模板变形，使其与网格对齐。请注意，在整个注册过程中，只有模板会变形，而目标网格仍保持固定状态。每次迭代都涉及以下四个步骤。

11 
11

1. Each vertex in the template mesh, vi ∈ V , is associated with a vertex, ci , on the target mesh, by evaluating the nearest neighbor in the embedding space. This step is different from the method described in [28], which computes the nearest neighbor in the Euclidean space. As a result, the proposed step allows registering a single template face to different facial identities with arbitrary expressions. 
  1.通过评估嵌入空间中的最近邻居，模板网格中的每个顶点vi∈V与目标网格上的顶点ci相关联。该步骤与[28]中描述的方法不同，后者计算欧几里德空间中的最近邻居。结果，所提出的步骤允许使用任意表达将单个模板面注册到不同的面部身份。

2. Pairs, (vi , ci ), which are physically distant by more than 1 millimeter and those with normal direction disagreement of more than 5 degrees are detected and ignored in the next step. 
  2.在下一步骤中检测并忽略物理距离超过1毫米的对（vi，ci）和具有超过5度的法线方向不一致的对。

3. The template mesh is deformed by minimizing the following energy 
  3.通过最小化以下能量使模板网格变形

E (V , C ) = αp2point 
E（V，C）\x3dαp2point

(cid:88) (cid:88) (cid:88) (cid:88) 
（cid：88）（cid：88）（cid：88）（cid：88）

(vi ,ci )∈J 
（vi，ci）∈J

i∈V 
i∈V

vj ∈N (vi ) 
vj∈N（vi）

(vi ,ci )∈J +αp2plane 
（vi，ci）∈J+αp2plane

+αmemb 
+αmemb

2 
2

(cid:107)vi − ci (cid:107)2 |(cid:126)n(ci )(vi − ci )|2 wi,j (cid:107)vi − vj (cid:107)2 2 , 
（cid：107）vi  -  ci（cid：107）2 |（cid：126）n（ci）（vi  -  ci）| 2 wi，j（cid：107）vi  -  vj（cid：107）2 2，

(8) 
（8）

where, wi,j is the weight corresponding to the biharmonic Laplacian operator (see [21, 5]), (cid:126)n(ci ) is the normal of the corresponding vertex at the target mesh ci , J is the set of the remaining associated vertex pairs (vi , ci ), and N (vi ) is the set 1-ring neighboring vertices about the vertex vi . Notice that the ﬁrst term above is the sum of squared Euclidean distances between matches and its weight αp2point is set to 0.1. The second term is the distance from the point vi to the tangent plane at the corresponding point on the target mesh, and its weight αp2plane is set to 1. The third term quantiﬁes the stiffness of the mesh and its weight αmemb is initialized to 108 . In practice, the energy term given in Equation 8 is minimized iteratively by an inner loop which contains a linear system of equations. We run this loop until the norm of the difference between the vertex positions of the current iteration and the previous one is below 0.01. 
其中，wi，j是对应于双调和拉普拉斯算子的权重（见[21,5]），（cid：126）n（ci）是目标网格ci对应顶点的法线，J是一组剩余的相关顶点对（vi，ci）和N（vi）是关于顶点vi的集合1环相邻顶点。请注意，上面的第一项是匹配之间的欧几里德距离的平方和，其权重αp2point设置为0.1。第二项是目标网格上对应点处从点vi到切平面的距离，其权重αp2plane设置为1.第三项量化网格的刚度，其权重αmemb初始化为108。在实践中，通过包含线性方程组的内环迭代地最小化等式8中给出的能量项。我们运行此循环，直到当前迭代的顶点位置与前一个迭代的顶点位置之间的差异的范数低于0.01。

4. If the motion of the template mesh between the current outer iteration and the previous one is below 0.1, we divide the weight αmemb by two. This relaxes the stiffness term and allows a greater deformation in the next outer iteration. In addition, we evaluate the difference between the number of remaining pairwise matches in the current iteration versus the previous one. If the difference is below 500, we modify the vertex association step to estimate the physical nearest neighbor vertex, instead of the the nearest neighbor in the space of the embedding given by the network. 
  4.如果模板网格在当前外部迭代和前一个迭代之间的运动低于0.1，我们将权重αmemb除以2。这放松了刚度项，并允许在下一次外迭代中更大的变形。此外，我们评估当前迭代中剩余的成对匹配数与前一个匹配数之间的差异。如果差异低于500，我们修改顶点关联步骤以估计物理最近邻居顶点，而不是网络给出的嵌入空间中的最近邻居。

This iterative process terminates when the stiffness weight αmemb is below 106 . The resulting output of this phase is a deformed template with ﬁxed triangulation, which contains the overall facial structure recovered by the network, yet, is smoother and complete. 
当刚度权重αmemb低于106时，该迭代过程终止。此阶段的结果输出是具有固定三角测量的变形模板，其包含由网络恢复的整体面部结构，但是更平滑和完整。

B.2. Fine Detail Reconstruction 
B.2。精细细节重建

Although the network already recovers ﬁne geometric details, such as wrinkles and moles, across parts of the face, a geometric approach can reconstruct details at a ﬁner level, on the entire face, independently of the resolution. Here, we propose an approach motivated by the passive-stereo facial reconstruction method suggested in [3]. The underlying assumption here is that subtle geometric structures can be explained by local variations in the image domain. For some skin tissues, such as nevi, this assumption is inaccurate as the intensity variation results from the albedo. In such cases, the geometric structure would be wrongly modiﬁed. Still, for most parts of the face, the reconstructed details are consistent with the actual variations in depth. The method begins from an interpolated version of the deformed template, provided by a surface subdivision technique. Each vertex v ∈ VD is painted with the intensity value of the nearest pixel in the image plane. Since we are interested in recovering small details, only the high spatial frequencies, µ(v), of the texture, τ (v), are taken into consideration in this phase. For computing this frequency band, we subtract the synthesized low frequencies from the original intensity values. This low-pass ﬁltered part can be computed by convolving the texture with a spatially varying Gaussian kernel in the image domain, as originally proposed. In contrast, since this convolution is equivalent to computing the heat distribution upon the shape after time dt, where the initial heat proﬁle is the original texture, we propose to compute µ(v) as 
虽然网络已经在面部的各个部分恢复了细微的几何细节，例如皱纹和痣，但是几何方法可以在整个面上的细节水平上重建细节，而与分辨率无关。在这里，我们提出了一种由[3]中提出的被动 - 立体面部重建方法推动的方法。这里的基本假设是细微的几何结构可以通过图像域中的局部变化来解释。对于一些皮肤组织，例如痣，这种假设是不准确的，因为强度变化是由反照率引起的。在这种情况下，几何结构将被错误地修改。尽管如此，对于脸部的大多数部分，重建的细节与深度的实际变化一致。该方法从变形模板的插值版本开始，由表面细分技术提供。每个顶点v∈VD用图像平面中最近像素的强度值绘制。由于我们对恢复小细节感兴趣，因此在该阶段仅考虑纹理的高空间频率μ（v）τ（v）。为了计算该频带，我们从原始强度值中减去合成的低频。如最初提出的，可以通过将纹理与图像域中的空间变化高斯核卷积来计算该低通滤波部分。相比之下，由于这个卷积相当于计算在时间dt之后的形状上的热分布，其中初始热量分布是原始纹理，我们建议将μ（v）计算为

µ(v) = τ (v) − (I − dt · ∆g )−1 τ (v), 
μ（v）\x3dτ（v） - （I  -  dt·Δg）-1τ（v），

(9) 
（9）

where I is the identity matrix, ∆g is the cotangent weight discrete Laplacian operator for triangulated meshes [31], and dt = 0.2 is a scalar proportional to the cut-off frequency of the ﬁlter. Next, we displace each vertex along its normal direction such that v (cid:48) = v + δ(v)(cid:126)n(v). The step size of the displacement, δ(v), is a combination of a data-driven term, δµ (v), and a regularization one, δs (v). The data-driven term is guided by the high-pass ﬁltered part of the texture, µ(v). In practice, we require the local differences in the geometry to be proportional to the local variation in the high frequency band of the texture. That is for each vertex v , with a normal (cid:126)n(v), and a neighboring vertex vi , the data-driven term is given by 
其中I是单位矩阵，Δg是三角网格的余切权重离散拉普拉斯算子[31]，而dt \x3d 0.2是与滤波器的截止频率成比例的标量。接下来，我们沿着其法线方向移动每个顶点，使得v（cid：48）\x3d v +δ（v）（cid：126）n（v）。位移的步长δ（v）是数据驱动项δμ（v）和正则化项δs（v）的组合。数据驱动的术语由纹理的高通滤波部分μ（v）引导。在实践中，我们要求几何中的局部差异与纹理的高频带中的局部变化成比例。这是针对每个顶点v，具有法线（cid：126）n（v）和相邻顶点vi，数据驱动项由下式给出：

(µ(v) − µ(vi )) = (cid:104)v + δµ (v)(cid:126)n(v) − vi , (cid:126)n(v)(cid:105). 
（μ（v）-μ（vi））\x3d（cid：104）v +δμ（v）（cid：126）n（v）-vi，（cid：126）n（v）（cid：105）。

Thus, the step size assuming a single neighboring vertex can be calculated by 
因此，可以通过计算假定单个相邻顶点的步长

δµ (v) = γ (µ(v) − µ(vi )) − (cid:104)v − vi , (cid:126)n(v)(cid:105). 
δμ（v）\x3dγ（μ（v）-μ（vi）） - （cid：104）v-vi，（cid：126）n（v）（cid：105）。

(10) 
（10）

(11) 
（11）

In the presence of any number of neighboring vertices of v , we compute the weighted average of its 1-ring neighborhood 
在存在任意数量的v的相邻顶点的情况下，我们计算其1环邻域的加权平均值

(cid:80) 
（CID：80）

δµ (v) = 
δμ（v）\x3d

vi∈N (v) α(v , vi )γ [(µ(v) − µ(vi )) − (cid:104)v − vi , (cid:126)n(v)(cid:105)] vi∈N (v) α(v , vi ) 
vi∈N（v）α（v，vi）γ[（μ（v） - μ（vi）） - （cid：104）v  -  vi，（cid：126）n（v）（cid：105）] vi∈N（v）α（v，vi）

(cid:80) 
（CID：80）

, 
，

(12) 
（12）

An alternative term can spatially attenuate the contribution of the data-driven term in curved regions for regularizing the reconstruction by 
另一个术语可以在空间上减弱数据驱动项在弯曲区域中的贡献，以使重建正规化

(cid:80) 
（CID：80）

vi∈N (v) 
vi∈N（v）

δµ (v) = 
δμ（v）\x3d

(cid:16) 
（CID：16）

(cid:17) 
（CID：17）

α(v ,vi ) (µ(v) − µ(vi )) 
α（v，vi）（μ（v） - μ（vi））

1 − |(cid:104)v−vi ,(cid:126)n(v)(cid:105)| 
1  -  |（cid：104）v-vi，（cid：126）n（v）（cid：105）|

(cid:107)v−vi (cid:107) 
（cid：107）v-vi（cid：107）

(cid:80) 
（CID：80）

vi∈N (v) 
vi∈N（v）

α(v ,vi ) 
α（v，vi）

, 
，

(13) 
（13）

where α(v ,vi ) = exp (−(cid:107)v − vi (cid:107)). where N (v) is the set 1-ring neighboring vertices about the vertex v , and (cid:126)n(v) is the unit normal at the vertex v . Since we move each vertex along the normal direction, triangles could intersect each other, particularly in regions with high curvature. To reduce the probability of such collisions, a regularizing displacement ﬁeld, δs (v), is added. This term is proportional to the mean curvature of the original surface, and is equivalent to a single explicit mesh fairing step [11]. The ﬁnal surface modiﬁcation is given by 
其中α（v，vi）\x3d exp（ - （cid：107）v  -  vi（cid：107））。其中N（v）是关于顶点v的集合1环相邻顶点，并且（cid：126）n（v）是顶点v处的单位法线。由于我们沿法线方向移动每个顶点，因此三角形可以相互交叉，特别是在具有高曲率的区域中。为了降低这种碰撞的可能性，增加了正则化位移场δs（v）。该项与原始曲面的平均曲率成比例，相当于单个显式网格整流步骤[11]。最终的表面修饰由下式给出

v (cid:48) = v + (ηδµ (v) + (1 − η)δs (v)) · (cid:126)n(v), 
v（cid：48）\x3d v +（ηδμ（v）+（1-η）δs（v））·（cid：126）n（v），

(14) 
（14）

for a constant η = 0.2. 
对于常数η\x3d 0.2。

C. Additional Experimental Results 
C.其他实验结果

We present additional qualitative results of our method. Figure 14 shows the output representations of the proposed network for a variety of different faces, notice the failure cases presented in the last two rows. One can see that the network generalizes well, but is still limited by the synthetic data. Speciﬁcally, the network might fail in presence of occlusions, facial hair or extreme poses. This is also visualized in Figure 15 where the correspondence error is visualized using the texture mapping. Additional reconstruction results of our method are presented in Figure 16. For analyzing the distribution of the error along the face, we present an additional comparison in Figure 17, where the absolute error, given in percents of the ground truth depth, is shown for several facial images. 
我们提出了我们方法的其他定性结果。图14显示了针对各种不同面的建议网络的输出表示，请注意最后两行中出现的故障情况。可以看出，网络概括良好，但仍然受到合成数据的限制。具体而言，网络可能在闭塞，面部毛发或极端姿势的情况下失败。这也在图15中可视化，其中使用纹理映射可视化对应误差。我们方法的其他重建结果如图16所示。为了分析沿面的误差分布，我们在图17中给出了一个额外的比较，其中显示了几个以地面实况深度的百分比给出的绝对误差。面部图像。

Figure 14: Network Output. 
图14：网络输出。

Figure 15: Results under occlusions and rotations. Input images are shown next to the matching correspondence result, visualized using the texture mapping to better show the errors. 
图15：遮挡和旋转下的结果。输入图像显示在匹配的对应结果旁边，使用纹理映射可视化以更好地显示错误。

Figure 16: Additional reconstruction results. 
图16：其他重建结果。

Figure 16: Additional reconstruction results. 
图16：其他重建结果。

Figure 16: Additional reconstruction results. 
图16：其他重建结果。

Input 
输入

Proposed 
建议

[34] 
[34]

[26] 
[26]

[49] 
[49]

Err. % Scale 
呃。 ％比例

Figure 17: Error heat maps in percentile of ground truth depth. 
图17：地面实况深度百分位数的误差热图。