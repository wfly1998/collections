# 大型姿势的面部对齐：3D解决方案

> Face Alignment Across Large Poses: A 3D Solution

$$
\rm{
Xiangyu\ Zhu^1\qquad Zhen\ Lei^{1*}\qquad Xiaoming\ Liu^2\qquad Hailin\ Shi^1\qquad Stan\ Z.\ Li^1
\\
^1Center\ for\ Biometrics\ and\ Security\ Research\ \&\ National\ Laboratory\ of\ Pattern\ Recognition,
\\
Institute\ of\ Automation,\ Chinese\ Academy\ of\ Sciences,
\\
^2Department\ of Computer\ Science\ and\ Engineering,\ Michigan\ State\ University
\\
\{xiangyu.zhu,zlei,hailin.shi,szli\}@nlpr.ia.ac.cn\qquad liuxm@msu.edu
}
$$

$^*通讯作者$

## 摘要

*面部对齐将面部模型与图像相对应，并提取面部像素的语义，是CV社区中的一个重要课题。但是，大多数算法都是针对小到中等姿势（低于$45^\circ$）的面部而设计的，缺乏在高达$90^\circ$的大型姿势中对齐面部的能力。挑战是三方面的：首先，常用的基于地标的人脸模型假设所有地标都是可见的，因此不适合于专业视图。其次，面部外观在大型姿势中变化更为显着，从正面视图到专业视图。第三，由于必须猜测无形的地标，因此大幅度标记地标极具挑战性。在本文中，我们提出了一种新的对齐框架中的三个问题的解决方案，称为3D密集人脸对齐（3DDFA），其中通过卷积中性网络（CNN）将密集的3D人脸模型配置到图像。我们还提出了一种在专业视图中合成大规模训练样本的方法，以解决数据标记的第三个问题。具有挑战性的AFLW数据库的实验表明，我们的方法比最先进的方法有了显着的改进。*

## 1.简介

传统的面部对齐旨在定位诸如“眼角”，“鼻尖”和“下巴中心”的面部特征点，基于该面部图像可以被标准化。它是许多面部分析任务的必要预处理步骤，例如，面部识别[41]，表情识别[5]和逆渲染[1]。面部对齐的研究可以分为两类：基于综合的分析[12,42,15]和基于回归的[11,17,27,45]。前者模拟图像生成过程，并通过最小化模型外观和输入图像之间的差异来实现对齐。后者提取关键点周围的特征，并将其回归到地面真相地标。随着过去十年的发展，在偏航角小于45°并且所有标志都可见的中等姿态下的面部对齐得到了很好的解决[45,51,54]。然而，在没有太多关注和成就的情况下，大面积（±90°）的面部对齐仍然是一个具有挑战性的问题。有三个主要挑战：

> 图1. 3DDFA的拟合结果。对于四个结果中的每一对，左边是具有平均纹理的嵌入3D形状的渲染，其被透明以证明装配精度。右侧是覆盖在3D人脸模型上的地标，其中蓝色/红色标志表示可见/不可见的地标。可见性直接由拟合的密集模型计算[21]。在补充材料中证明了更多结果。

**建模**：地标形状模型[13]隐含地假设每个地标可以基于其独特的视觉模式被鲁棒地检测。然而，当面部偏离正面视图时，由于自我遮挡，一些标志变得不可见[53]。在中等姿势中，可以通过将面部轮廓标志的语义位置改变为轮廓来解决该问题，这被称为地标行进[55]。然而，在一半脸部被遮挡的大型姿势中，一些地标不可避免地不可见并且没有图像数据。结果，地标形状模型不再有效。

**适合**：大型姿势的面部对齐比中型姿势更具挑战性，因为当靠近轮廓视图时具有戏剧性的外观146变化。 级联线性回归[45]或传统非线性模型[27,50,10]不足以以统一的方式覆盖此类复杂模式。 基于视图的框架采用不同的地标配置和每个视图类别[53,49,56,38]的拟合模型，可能会显着增加计算成本，因为每个视图都必须进行测试。

**数据标记**：最严重的问题来自数据。在大型人脸上手动标记地标是非常繁琐的，因为必须“猜测”被遮挡的地标，这对大多数人来说是不可能的。因此，大多数公共面部对齐数据库，如AFW [56]，LFPW [22]，HELEN [26]和IBUG [35]都以中等姿势收集。现有的大型姿势数据库（如AFLW [25]）仅包含可见的地标，这些地标在不可见的地标中可能是模糊的，并且很难训练单一的面部对齐模型。在本文中，我们解决了所有三个挑战，目标是改善大型姿势的面部对齐性能。

1. 为了解决大型姿势中不可见地标的问题，我们建议将3D密集人脸模型而不是稀疏地标形状模型应用于图像。通过结合3D信息，可以固有地解决由3D变换引起的外观变化和自遮挡。我们称这种方法为3D Dense Face Alignment（3DDFA）。一些结果如图1所示。
2. 为了解决3DDFA中的设置过程，我们提出了一种基于级联卷积中性网络（CNN）的回归方法。 CNN已经被证明具有从物体检测[48]和图像分类[40]中具有大变化的图像中提取有用信息的出色能力。在这项工作中，我们采用CNN来设计具有特定设计特征的3D人脸模型，即投影归一化坐标代码（PNCC）。此外，提出加权参数距离成本（WPDC）作为成本函数。据我们所知，这是首次尝试解决与CNN的3D人脸对齐问题。
3. 为了实现3DDFA的训练，我们构建了包含2D人脸图像和3D人脸模型的人脸数据库。我们进一步提出了一种面部验证算法来合成大型姿势的60k +训练样本。合成的样本很好地模拟了大型姿势中的面部外观，并提高了先前和我们提出的面部对齐算法的性能。

数据库和代码在[http://www.cbsr.ia.ac.cn/users/xiangyuzhu/](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/)发布。

## 2. 相关工作

**通用面部对齐**：2D中的面部对齐旨在定位一组稀疏的面部地标。已经取得了许多成就，包括经典的主动外观模型（AAM）[12,36,42]和约束局部模型（CLM）[16,37,2]。最近，已经提出了基于回归的方法，其将地标周围的辨别特征映射到期望的界标位置[43,45,46,10,50,27]。通过利用回归的输出（界标位置）对输入（地标上的特征）影响的反馈特性，级联回归[17]级联弱回归量列表，逐步减少对齐误差并达到状态艺术[46,54]。

除了传统模型，卷积中性网络（CNN）最近也被用于面部对齐。孙等人。 [39]首先使用CNN用原始面部图像回归地标位置。梁等人。 [28]通过估计地标响应图来提高灵活性。张等人。 [51]通过多任务CNN进一步将面部对齐与属性分析相结合，以提高两个任务的性能。尽管取得了相当大的成就，但大多数CNN方法只能检测到一组稀疏的标志性建筑（[39,51,28]中的5个点），其表面形状的描述能力有限。

**大姿势脸部对齐**：尽管人脸对齐非常受关注，但大型姿势的文献相当有限。最常见的方法是多视图框架[14]，它为不同的视图使用不同的地标配置。例如，TSPM [56]和CDM [49]采用类似DPM的[18]方法来对齐具有不同形状模型的面部，其中选择最高可能性作为最终结果。但是，由于必须测试每个视图，因此多视图方法的计算成本总是很高。

除了2D方法之外，3D面部对齐[19]通过最小化图像和模型外观之间的差异来建立3D可变形模型（3DMM）[6]，也有可能处理大的姿势[6,33]。但是，它受到每图像一分钟计算成本的影响。最近，已经提出了基于回归的3DMM配置，其通过回归地标位置[49,24,8,23]处的特征来估计模型参数，以提高效率。然而，由于地标上的特征可能如2D方法那样自遮挡，因此装配算法不再是姿势不变的。

## 3. 3D密集面对齐（3DDFA）

在本节中，我们将介绍3D Dense Face Alignment（3DDFA），它具有级联CNN的3D可变形模型。

> 图2. 3DDFA概述在第k次迭代中，Netk将中间参数p k作为输入，构造投影的归一化k。坐标代码（PNCC），将其与输入图像堆叠并发送到CNN以预测参数更新Δp

### 3.1 3D变形模型

Blanz等人[6]提出了使用PCA描述3D面部空间的3D可变形模型（3DMM）：

$$
\mathbf S=\overline{\mathbf S}+\mathbf A_{id}\alpha_{id}+\mathbf A_{exp}\alpha_{exp},\tag1
$$
其中$\mathbf S$是3D面，$\overline{\mathbf S}$是平均形状，$\mathbf A_{id}$是在3D面部扫描上训练的主轴，具有中性表达式，$\alpha_{id}$是形状参数，$\mathbf A_{exp}$是在表达扫描和中性扫描之间的偏移上训练的主轴。 $\alpha_{exp}$是表达式参数。在这项工作中，$\mathbf A_{id}$和$\mathbf A_{exp}$分别来自BFM [31]和FaceWarehouse [9]。然后使用弱透视投影将3D面投影到图像平面上：
$$
V(\mathbf p)=f*\mathbf{Pr}∗\mathbf R*(\overline{\mathbf S}+\mathbf A_{id}\alpha_{id}+\mathbf A_{exp}\alpha_{exp})+\mathbf t_{2d},\tag2
$$
其中$V(\mathbf p)$是模型构造和投影函数，导致模型顶点的2D位置，$f$是比例因子，$\mathbf {Pr}$是正交投影矩阵$\left(\begin{matrix}1&0&0\\0&1&1\end{matrix}\right)$，$\mathbf R$是从中构造的旋转矩阵旋转角度俯仰，偏航，滚动和$t_{2d}$是平移向量。所有模型参数的集合是$\mathbf p=[f, pitch, yaw, roll, \mathbf t_{2d}, \alpha_{id}, \alpha_{exp}]^T$。

### 3.2 网络结构

3D面部对齐的目的是从单面图像$\mathbf I$估计$\mathbf p$。与现有的CNN方法[39,28]不同，3DDFA在不同的配置阶段应用不同的网络，3DDFA在级联中采用统一的网络结构。一般来说，在迭代$k\ (k=0,1,\dots,K)$，给定初始参数$\mathbf p^k$，我们用$\mathbf p^k$构造一个特殊设计的特征PNCC，并训练卷积中性网络$Net^k$来预测参数更新$\Delta\mathbf p^k$：

$$
\Delta\mathbf p^k=Net^k(\mathbf I,\rm{PNCC(\mathbf p^k)}),\tag3
$$
之后，更好的中间参数$\mathbf p^{k+1}=\mathbf p^k+\Delta\mathbf p^k$成为下一个网络$Net^{k+1}$的输入，其具有与$Net^k$相同的结构。 图2显示了网络结构。输入是由PNCC堆叠的$100\times100\times3$彩色图像。该网络包含四个卷积层，三个池化层和两个完全连接的层。前两个卷积层共享权重以提取低级特征。最后两个卷积层不共享权重以提取位置敏感特征，其进一步回归到256维特征向量。输出是234维参数更新，包括6维姿势参数[f，俯仰，偏航，滚动，t2dx，t2dy]，199维形状参数αid和29维表达参数

αexp . 
αexp。

3.3. Projected Normalized Coordinate Code 
3.3。预计的归一化坐标代码

The special structure of the cascaded CNN has three requirements of its input feature: Firstly, the feedback property requires that the input feature should depend on the CNN output to enable the cascade manner. Secondly, the convergence property requires that the input feature should reﬂect the ﬁtting accuracy to make the cascade converge after some iterations [57]. Finally, the convolvable property requires that the convolution on the input feature should make sense. Based on the three properties, we 
级联CNN的特殊结构对其输入特性有三个要求：首先，反馈特性要求输入特性应取决于CNN输出以启用级联方式。其次，收敛性要求输入特征应反映拟合精度，以使级联在一些迭代后收敛[57]。最后，可卷曲属性要求输入要素上的卷积才有意义。基于这三个属性，我们

(a) NCC 
（a）NCC

(b) PNCC 
（b）PNCC

Figure 3. The Normalized Coordinate Code (NCC) and the Projected Normalized Coordinate Code (PNCC). (a) The normalized mean face, which is also demonstrated with NCC as its texture (NCCx = R, NCCy = G, NCCz = B). (b) The generation of PNCC: The projected 3D face is rendered by Z-Buffer with NCC as its colormap. 
图3.标准化坐标代码（NCC）和预计标准化坐标代码（PNCC）。 （a）归一化的平均面，也用NCC作为其纹理证明（NCCx \x3d R，NCCy \x3d G，NCCz \x3d B）。 （b）PNCC的生成：投影的3D面由Z-Buffer渲染，NCC作为其颜色图。

Afterwards, a better medium parameter pk+1 = pk + ∆pk becomes the input of the next network Netk+1 which has the same structure as Netk . Fig. 2 shows the network 
之后，更好的中间参数pk + 1 \x3d pk +Δpk成为下一个网络Netk + 1的输入，其具有与Netk相同的结构。图2显示了网络

design our features as follows: Firstly, the 3D mean face is normalized to 0 − 1 in x, y , z axis as Equ. 4. The unique 3D coordinate of each vertex is called its Normalized Coor
设计我们的特征如下：首先，将3D平均面在x，y，z轴上标准化为0-1为Equ。 4.每个顶点的唯一3D坐标称为标准化Coor

148 
148

dinate Code (NCC), see Fig. 3(a). 
dinate Code（NCC），见图3（a）。

NCCd = 
NCCd \x3d

Sd − min(Sd ) max(Sd ) − min(Sd ) 
Sd  -  min（Sd）max（Sd） -  min（Sd）

(d = x, y , z ), 
（d \x3d x，y，z），

(4) 
（4）

where the S is the mean shape of 3DMM in Equ. 1. Since NCC has three channels as RGB, we also show the mean face with NCC as its texture. Secondly, with a model parameter p, we adopt the Z-Buffer to render the projected 3D face colored by NCC as in Equ. 5, which is called the Projected Normalized Coordinate Code (PNCC), see Fig. 3(b): 
其中S是Equ中3DMM的平均形状。 1.由于NCC有三个通道作为RGB，我们还显示了NCC作为其纹理的平均面。其次，使用模型参数p，我们采用Z-Buffer来渲染由NCC着色的投影3D面部，如同Equ。 5，称为投影归一化坐标代码（PNCC），见图3（b）：

PNCC = Z-Buffer(V3d (p), NCC) 
PNCC \x3d Z缓冲区（V3d（p），NCC）

V3d (p) = f ∗ R ∗ S + [t2d , 0]T 
V3d（p）\x3d f * R * S + [t2d，0] T.

(5) 
（5）

S = S + Aidαid + Aexpαexp , 
S \x3d S +Aidαid+Aexpαexp，

where Z-Buffer(ν , τ ) renders an image from the 3D mesh ν colored by τ and V3d (p) is the current 3D face. Afterwards, PNCC is stacked with the input image and transferred to CNN. Regarding the three properties, PNCC fulﬁlls the feedback property since in Equ. 5, p is the output of CNN and NCC is a constant. Secondly, PNCC provides the 2D locations of visible 3D vertexes on the image plane. When CNN detects that each NCC superposes its corresponding image pattern during testing, the cascade will converge. PNCC fulﬁlls the convergence property. Note that the invisible region is automatically ignored by Z-Buffer. Finally, PNCC is smooth in 2D space, the convolution indicates the linear combination of NCCs on a local patch. It fulﬁlls the convolvable property. 
其中Z-Buffer（ν，τ）渲染来自由τ着色的3D网格ν的图像，并且V3d（p）是当前3D面部。之后，PNCC与输入图像堆叠并传输到CNN。关于这三个属性，PNCC满足了Equ的反馈属性。 5，p是CNN的输出，NCC是常数。其次，PNCC在图像平面上提供可见3D顶点的2D位置。当CNN检测到每个NCC在测试期间叠加其对应的图像模式时，级联将收敛。 PNCC满足了收敛性。请注意，Z-Buffer会自动忽略不可见区域。最后，PNCC在2D空间中是平滑的，卷积表示NCC在本地补丁上的线性组合。它充满了可卷曲的财产。

3.4. Cost Function 
3.4。成本函数

The performance of CNN can be greatly impacted by the cost function, which is difﬁcult to design in 3DDFA since each dimension of the CNN output (model parameter) has different inﬂuence on the 3DDFA result (ﬁtted 3D face). In this work, we discuss two baselines and propose a novel cost function. Since the parameter range varies signiﬁcantly, we conduct z-score normalization before training. 
由于CNN输出的每个维度（模型参数）对3DDFA结果（具有3D面部）的影响不同，因此在3DDFA中难以设计成本函数，因此CNN的性能会受到很大影响。在这项工作中，我们讨论了两个基线并提出了一种新的成本函数。由于参数范围变化很​​大，我们在训练前进行z分数归一化。

3.4.1 Parameter Distance Cost (PDC) 
3.4.1参数距离成本（PDC）

Take the ﬁrst iteration as an example. The purpose of CNN is predicting the parameter update ∆p to move the initial parameter p0 closer to the ground truth pg . Intuitively, we can minimize the distance between the ground truth and the current parameter with the Parameter Distance Cost (PDC): 
以第一次迭代为例。 CNN的目的是预测参数更新Δp以使初始参数p0更接近地面实况pg。直观地说，我们可以使用参数距离成本（PDC）最小化地面实况与当前参数之间的距离：

Epdc = k∆p − (pg − p0 )k2 . 
Epdc \x3dkΔp - （pg-p0）k2。

(6) 
（6）

Even though PDC has been used in 3D face alignment [57], there is a problem that each dimension in p has different inﬂuence on the resultant 3D face. For example, with the 
尽管PDC已用于3D人脸对齐[57]，但存在一个问题，即p中的每个维度对所得到的3D脸部的影响不同。例如，随着

same deviation, the yaw angle will bring a larger alignment error than a shape PCA coefﬁcient, while PDC optimizes them equally. 
相同的偏差，偏航角将带来比形状PCA系数更大的对准误差，而PDC同样优化它们。

3.4.2 Vertex Distance Cost (VDC) 
3.4.2顶点距离成本（VDC）

Since 3DDFA aims to morph the 3DMM to the ground truth 3D face, we can optimize ∆p by minimizing the vertex distances between the ﬁtted and the ground truth 3D face: 
由于3DDFA旨在将3DMM变形为地面真实3D面部，我们可以通过最小化拼接和地面真实3D面之间的顶点距离来优化Δp：

Evdc = kV (p0 + ∆p) − V (pg )k2 , 
Evdc \x3d kV（p0 +Δp） -  V（pg）k2，

(7) 
（7）

where V (·) is the face construction and weak perspective projection as Equ. 2. This cost is called the Vertex Distance Cost (VDC) and the derivative is provided in supplemental material. Compared with PDC, VDC better models the ﬁtting error by explicitly considering the semantics of each parameter. However, we observe that VDC exhibits pathological curvature [29]. The directions of pose parameters always exhibit much higher curvatures than the PCA coefﬁcients. As a result, optimizing VDC with gradient descend converges very slowly due to the “zig-zagging” problem. Second-order optimizations are preferred but they are expensive and hard to be implemented on GPU. 
其中V（·）是面部构造，弱视角投影为Equ。 2.此成本称为顶点距离成本（VDC），衍生物以补充材料的形式提供。与PDC相比，VDC通过明确考虑每个参数的语义，更好地模拟了配置错误。然而，我们观察到VDC表现出病理性曲率[29]。姿势参数的方向总是表现出比PCA系数高得多的曲率。因此，由于“锯齿形”问题，优化具有梯度下降的VDC会非常缓慢地收敛。二阶优化是首选，但它们很昂贵且难以在GPU上实现。

3.4.3 Weighted Parameter Distance Cost (WPDC) 
3.4.3加权参数距离成本（WPDC）

In this work, we propose a simple but effective cost function Weighted Parameter Distance Cost (WPDC). The basic idea is explicitly modeling the importance of each parameter: 
在这项工作中，我们提出了一个简单但有效的成本函数加权参数距离成本（WPDC）。基本思想是明确建模每个参数的重要性：

Ewpdc = (∆p − (pg − p0 ))T W(∆p − (pg − p0 )) where W = diag(w1 ,w2 , ..., wn ) wi = kV (pd (i))−V (pg )k/ X wi pd (i)i = (p0 + ∆p)i 
Ewpdc \x3d（Δp - （pg-p0））TW（Δp - （pg-p0））其中W \x3d diag（w1，w2，...，wn）wi \x3d kV（pd（i）） -  V（ pg）k / X wi pd（i）i \x3d（p0 +Δp）i

pd (i)j = pg 
pd（i）j \x3d pg

j , 
j，

j ∈ {1, . . . , i − 1, i + 1, . . . , n}, 
j∈{1 ,. 。 。 ，我 -  1，我+ 1 ,. 。 。 ，n}，

(8) 
（8）

where W is the importance matrix whose diagonal is the weight of each parameter, pd (i) is the i-deteriorated parameter whose ith component comes from the predicted parameter (p0 + ∆p) and the others come from the ground truth parameter pg , kV (pd (i)) − V (pg )k models the alignment error brought by miss-predicting the ith model parameter, which is indicative of its importance. For simplicity, W is considered as a constant when computing the derivative. In the training process, CNN ﬁrstly concentrate on the parameters with larger kV (pd (i)) − V (pg )k such as scale, rotation and translation. As pd (i) is closer to pg , the weights of these parameters begin to shrink and CNN will optimize less important parameters but at the same time keep the high-priority parameters sufﬁciently good. Compared with VDC, the WPDC remedies the pathological curvature issue and is easier to implement without the derivative of V (·). 
其中W是重要性矩阵，其对角线是每个参数的权重，pd（i）是i-恶化参数，其第i个分量来自预测参数（p0 +Δp），其他参数来自地面实况参数pg， kV（pd（i）） -  V（pg）k模拟由于错误预测第i个模型参数而带来的对准误差，这表示其重要性。为简单起见，在计算导数时，W被视为常数。在训练过程中，CNN首先关注具有较大kV（pd（i）） -  V（pg）k的参数，例如比例，旋转和平移。随着pd（i）更接近pg，这些参数的权重开始缩小，CNN将优化不太重要的参数，但同时保持高优先级参数足够好。与VDC相比，WPDC可以弥补病理曲率问题，并且在没有V（·）导数的情况下更容易实现。

149 
149

4. Face Proﬁling 
  4.面对专业人士

All the discriminative models rely on the training data, especially for CNN which has thousands of parameters to train. Therefore, massive labelled faces across large poses are crucial for 3DDFA. However, few of released face alignment database contains large-pose samples [56, 22, 26, 35] since labelling standardized landmarks on proﬁle is very challenging. In this section, we demonstrate that labelled proﬁle faces can be well simulated from existing training samples with the help of 3D information. Inspired by the recent breakthrough in face frontalization [55, 21] which generates the frontal view of faces, we propose to invert this process to generate the proﬁle view of faces from mediumpose samples, which is called face proﬁling. The basic idea is predicting the depth of face image and generating the proﬁle views with 3D rotation. 
所有判别模型都依赖于训练数据，特别是对于有数千个训练参数的CNN。因此，大型姿势的大量标记面对3DDFA至关重要。然而，很少发布的面部对齐数据库包含大型样本[56,22,26,35]，因为在专业文章上标记标准化地标非常具有挑战性。在本节中，我们证明了借助于3D信息可以从现有的训练样本中很好地模拟标记的配置面。受到最近面部正面化[55,21]的突破的启发，我们建议将这一过程转化为生成人体样本中面部的专业视图，称为面部特征。基本思想是预测人脸图像的深度并通过3D旋转生成专业视图。

4.1. 3D Image Meshing 
4.1。 3D图像网格划分

The depth estimation of a face image can be conducted on the face region and external region respectively, with different requirements of accuracy. On the face region, we ﬁt a 3DMM through the Multi-Features Framework [33] (MFF), see Fig. 4(b). With the ground truth landmarks as a solid constraint throughout the ﬁtting process, the MFF can always converge to a very good result. Few failed samples can be easily adjusted manually. On the external region, we follow the 3D meshing method proposed by Zhu et al. [55] to mark some anchors beyond the face region and estimate their depth, see Fig. 4(c). Afterwards the whole image is tuned into a 3D object through triangulation, see Fig. 4(c)4(d). 
可以分别在面部区域和外部区域上进行面部图像的深度估计，具有不同的精度要求。在面部区域，我们通过多特征框架[33]（MFF）得到3DMM，见图4（b）。在整个装配过程中，地面实况标志作为一个坚实的约束，MFF总能收敛到一个非常好的结果。很少有失败的样品可以轻松手动调整。在外部区域，我们遵循Zhu等人提出的3D网格划分方法。 [55]标记一些超出面部区域的锚点并估计它们的深度，见图4（c）。然后通过三角测量将整个图像调谐成3D对象，参见图4（c）4（d）。

(a) 
（一个）

(b) 
（b）中

(c) 
（C）

(d) 
（d）

Figure 4. 3D Image Meshing. (a) The input image. (b) The ﬁtted 3D face through MFF. (c) The depth image from 3D meshing. (d) A different view of the depth image. 
图4. 3D图像网格划分。 （a）输入图像。 （b）通过MFF安装3D面部。 （c）3D网格划分的深度图像。 （d）深度图像的不同视图。

4.2. 3D Image Rotation 
4.2。 3D图像旋转

When the depth information is estimated, the face image can be rotated in 3D space to generate the appearances in larger poses (Fig. 5). It can be seen that the external face region is necessary for a realistic proﬁle image. Different from face frontalization, with larger rotation angles the self-occluded region can only be expanded. As a result, we avoid the troubling invisible region ﬁlling which may produce large artifacts [55]. 
当估计深度信息时，可以在3D空间中旋转面部图像以产生更大姿势的外观（图5）。可以看出，外部面部区域对于逼真的专业图像是必需的。与面部正面化不同，具有较大的旋转角度，自遮挡区域只能扩展。因此，我们避免了可能产生大量伪影的令人不安的无形区域填充[55]。

(a) 
（一个）

(b) 
（b）中

(c) 
（C）

(d) 
（d）

Figure 5. 2D and 3D view of the image rotation. (a) The original yaw angle yaw0 . (b) yaw0 +20◦ . (c) yaw0 +30◦ . (d) yaw0 +40◦ . 
图5.图像旋转的2D和3D视图。 （a）原始偏航角yaw0。 （b）yaw0 + 20°。 （c）yaw0 + 30°。 （d）yaw0 + 40°。

In this work, we enlarge the yaw of the depth image at the step of 5◦ until 90◦ . Through face proﬁling, we not only obtain the face appearances in large poses and but also augment the dataset to a large scale, which means the CNN can be well trained even given a small database. 
在这项工作中，我们在5°到90°的步长处放大深度图像的偏航。通过面部验证，我们不仅可以获得大型姿势的脸部外观，而且还可以大规模地增加数据集，这意味着即使给定小型数据库，CNN也可以得到很好的训练。

5. Implementation Details 
  5.实施细节

5.1. Initialization Regeneration 
5.1。初始化再生

With a huge number of parameters, CNN tends to overﬁt the training set and the networks at deeper cascade might receive training samples with almost zero errors. Therefore we cannot directly adopt the cascade framework as in 2D face alignment. Asthana et al. [3] demonstrates that the initializations at each iteration can be well simulated with statistics. In this paper, we also regenerate the pk but with a more sophisticated method. We observe that the ﬁtting error highly depends on the ground truth face posture (FP). For example, the error of a proﬁle face is mostly caused by a small yaw angle and the error of an open-mouth face is always caused by a close-mouth expression parameter. As a result, it is reasonable to model the perturbation of a training sample with a set of similar-FP samples. In this paper, we deﬁne the face posture as the ground truth 2D landmarks without scale and translation: 
由于具有大量参数，CNN倾向于超过训练集，并且更深级联的网络可能接收具有几乎零误差的训练样本。因此，我们不能像2D面对齐那样直接采用级联框架。 Asthana等。 [3]表明，每次迭代的初始化都可以用统计数据很好地模拟。在本文中，我们还使用更复杂的方法重新生成pk。我们观察到，装配误差高度依赖于地面实况面部姿势（FP）。例如，配置面的误差主要是由小偏航角引起的，并且张开嘴面的误差总是由闭口表达参数引起。因此，使用一组类似的FP样本对训练样本的扰动进行建模是合理的。在本文中，我们将面部姿势定义为没有尺度和平移的地面真相2D地标：

FP = Pr ∗ Rg ∗ (S + Aidα 
FP \x3d Pr * Rg *（S +Aidα

g id + Aexpα 
g id +Aexpα

g exp )landmark , (9) 
g exp）landmark，（9）

g id , αg 
g id，αg

where Rg , α exp represent the ground truth pose, shape and expression respectively and the subscript landmark means only landmark points are selected. Before training, we select two folds of samples as the validation set. For each training sample, we construct a validation subset {v1 , ..., vm } whose members share similar FP with the training sample. At iteration k , we regenerate the initial parameter by: 
其中Rg，αexp分别代表地面真实姿态，形状和表达，下标地标意味着只选择地标点。在训练之前，我们选择两个样本作为验证集。对于每个训练样本，我们构建验证子集{v1，...，vm}，其成员与训练样本共享相似的FP。在迭代k，我们通过以下方式重新生成初始参数：

pk = pg − (pg 
pk \x3d pg  - （pg

vi − pk 
vi  -  pk

vi ), 
vi），

(10) 
（10）

150 
150

where pk and pg are the initial and ground truth parameter of a training sample, pk vi and pg vi come from a validation sample vi which is randomly chosen from the corresponding validation subset. Note that vi is never used in training. 
其中pk和pg是训练样本的初始和基础真实参数，pk vi和pg vi来自验证样本vi，其从相应的验证子集中随机选择。请注意，vi从未在训练中使用过。

5.2. Landmark Reﬁnement 
5.2。地标改造

Dense face alignment method ﬁts all the vertexes of the face model by estimating the model parameters. If we are only interested in a sparse set of points such as landmarks, the error can be further reduced by relaxing the PCA constraint. In the 2D face alignment task, after 3DDFA we extract HOG features at landmarks and train a linear regressor to reﬁne the landmark locations. In fact, 3DDFA can team with any 2D face alignment methods. In the experiment, we also report the results reﬁned by SDM [45]. 
密集面部对齐方法通过估计模型参数来处理面部模型的所有顶点。如果我们只对诸如地标之类的稀疏点感兴趣，则可以通过放宽PCA约束来进一步减少误差。在2D面部对齐任务中，在3DDFA之后，我们在地标处提取HOG特征并训练线性回归器以重新定义地标位置。事实上，3DDFA可以与任何2D面部对齐方法结合使用。在实验中，我们还报告了SDM [45]重新定义的结果。

6. Experiments 
  6.实验

In this section, we evaluate the performance of 3DDFA in three common face alignment tasks in the wild, i.e., medium-pose face alignment, large-pose face alignment and 3D face alignment. Due to the space constraint, qualitative alignment results are shown in supplemental material. 
在本节中，我们将评估3DDFA在野外三种常见面部对齐任务中的性能，即中等姿势面部对齐，大型姿势面部对齐和3D面部对齐。由于空间限制，定性对齐结果显示在补充材料中。

6.1. Datasets 
6.1。数据集

Evaluations are conducted with three databases, 300W [34], AFLW [25] and a speciﬁcally constructed AFLW2000-3D database. 300W-LP: 300W [34] standardises multiple alignment databases with 68 landmarks, including AFW [56], LFPW [4], HELEN [52], IBUG [34] and XM2VTS [30]. With 300W, we adopt the proposed face proﬁling to generate 61,225 samples across large poses (1,786 from IBUG, 5,207 from AFW, 16,556 from LFPW and 37,676 from HELEN, XM2VTS is not used), which is further expanded to 122,450 samples with ﬂipping. We call the database as the 300W across Large Poses (300W-LP) AFLW: AFLW [25] contains 21,080 in-the-wild faces with large-pose variations (yaw from −90◦ to 90◦ ). Each image is annotated with up to 21 visible landmarks. The dataset is very suitable for evaluating face alignment performance across large poses. AFLW2000-3D: Evaluating 3D face alignment in the wild is difﬁcult due to the lack of pairs of 2D image and 3D model in unconstrained environment. Considering the recent achievements in 3D face reconstruction which can construct a 3D face from 2D landmarks [1, 44, 55, 20], we assume that a 3D model can be accurately ﬁtted if sufﬁcient 2D landmarks are provided. Therefore 3D evaluation can be degraded to 2D evaluation which also makes it possible to compare 3DDFA with other 2D face alignment methods. However, AFLW is not suitable for evaluating this task since only visible landmarks lead to serious ambiguity in 3D 
使用三个数据库进行评估，300W [34]，AFLW [25]和特定构建的AFLW2000-3D数据库。 300W-LP：300W [34]标准化具有68个界标的多个比对数据库，包括AFW [56]，LFPW [4]，HELEN [52]，IBUG [34]和XM2VTS [30]。使用300W，我们采用所提出的面部轮廓来生成大型姿势的61,225个样本（来自IBUG的1,786个，来自AFW的5,207个，来自LFPW的16,556个和来自HELEN的37,676个，不使用XM2VTS），其进一步扩展到122,450个样本。我们将数据库称为300W横跨大姿势（300W-LP）AFLW：AFLW [25]包含21,080个具有大姿势变化的野外面（偏航从-90°到90°）。每张图片都注释了多达21个可见的地标。该数据集非常适合评估大型姿势的面部对齐性能。 AFLW2000-3D：由于在无约束环境中缺少成对的2D图像和3D模型，因此评估野外3D面部对齐是困难的。考虑到最近在3D人脸重建方面取得的成就，可以从2D地标[1,44,55,20]构建3D人脸，我们假设如果提供了足够的2D地标，则可以准确地确定3D模型。因此，3D评估可以降级为2D评估，这也使得可以将3DDFA与其他2D面部对齐方法进行比较。但是，AFLW不适合评估此任务，因为只有可见的地标会导致3D严重模糊

shape, as reﬂected by the fake good alignment phenomenon in Fig. 6. In this work, we construct a database called AFLW2000-3D for 3D face alignment evaluation, which contains the ground truth 3D faces and the corresponding 68 landmarks of the ﬁrst 2,000 AFLW samples. Construction details are provided in supplemental material. 
由图6中假的良好对齐现象反映出来的形状。在这项工作中，我们构建了一个名为AFLW2000-3D的数据库，用于3D面部对齐评估，其中包含地面实况3D面和第一个2,000 AFLW的相应68个地标。样本。施工细节以补充材料提供。

Figure 6. Fake good alignment in AFLW. For each sample, the ﬁrst shows the visible 21 landmarks and the second shows all the 68 landmarks. The Normalized Mean Error (NME) reﬂects their accuracy. It can be seen that only evaluating visible landmarks cannot well reﬂect the ﬁtting accuracy. 
图6. AFLW中的假良好对齐。对于每个样本，第一个显示可见的21个地标，第二个显示所有68个地标。归一化均值误差（NME）反映了它们的准确性。可以看出，仅评估可见地标不能很好地反映拟合精度。

6.2. Performance Analysis 
6.2。绩效分析

Error Reduction in Cascade: To analyze the error reduction process in cascade and evaluate the effect of initialization regeneration. We divide 300W-LP into 97,967 samples for training and 24,483 samples for testing, without identity overlapping. Fig. 7 shows the training and testing errors at each iteration, with and without initialization regeneration. As observed, the testing error is reduced due to 
级联中的错误减少：分析级联中的错误减少过程并评估初始化重新生成的效果。我们将300W-LP划分为97,967个样本用于训练，24,483个样本用于测试，没有身份重叠。图7显示了每次迭代时的训练和测试错误，有和没有初始化再生。如所观察到的，测试错误由于减少而减少

(a) 
（一个）

(b) 
（b）中

Figure 7. The training and testing errors with (a) and without (b) initialization regeneration. 
图7.训练和测试错误（a）和没有（b）初始化再生。

initialization regeneration. In the generic cascade process the training and testing errors converge fast after 2 iterations. While with initialization regeneration, the training error is updated at the beginning of each iteration and the testing error continues to descend. During testing, 3DDFA takes 25.24ms for each iteration, 17.49ms for PNCC construction on 3.40GHZ CPU and 7.75ms for CNN on GTX TITAN Black GPU. Note that the computing time of PNCC can be greatly reduced if Z-Buffer is conducted on GPU. Considering both effectiveness and efﬁciency we choose 3 iterations in 3DDFA. Performance with Different Costs: In this experiment, we demonstrate the performance with different costs including PDC, VDC and WPDC. Fig. 8 demonstrates the testing 
初始化再生。在通用级联过程中，训练和测试误差在2次迭代后快速收敛。在初始化重新生成时，训练错误在每次迭代开始时更新，测试错误继续下降。在测试期间，每次迭代3DDFA需要25.24ms，在3.40GHZ CPU上需要17.49ms用于PNCC构建，在GTX TITAN Black GPU上用于CNN需要7.75ms。请注意，如果在GPU上进行Z-Buffer，则可以大大降低PNCC的计算时间。考虑到有效性和效率，我们在3DDFA中选择3次迭代。不同成本的性能：在本实验中，我们以不同的成本演示了性能，包括PDC，VDC和WPDC。图8展示了测试

151 
151

errors at each iteration. All the networks are trained until convergence. It is shown that PDC cannot well model the 
每次迭代时的错误。所有网络都经过训练直到收敛。结果表明PDC不能很好地模拟

results and Fig. 9 shows the corresponding CED curves. Each method is trained on 300W and 300W-LP respectively to demonstrate the boost from face proﬁling. If a trained model is provided in the code, we also demonstrate its performance. Since CDM only contains testing code, we just report its performance with the provided alignment model. For 3DDFA which depends on large scales of data, we only report its performance trained on 300W-LP. 
结果和图9显示了相应的CED曲线。每种方法分别在300W和300W-LP上进行训练，以展示面部优化的提升。如果代码中提供了训练有素的模型，我们也会展示其性能。由于CDM仅包含测试代码，因此我们只使用提供的对齐模型报告其性能。对于依赖于大​​规模数据的3DDFA，我们仅报告其在300W-LP上训练的性能。

Figure 8. The testing errors with different cost function. 
图8.具有不同成本函数的测试错误。

ﬁtting error and converges to an unsatisﬁed result. VDC is better than PDC, but the pathological curvature problem makes it only concentrate on a small set of parameters, which limits its performance. WPDC explicitly models the priority of each parameter and adaptively optimizes them with the parameter weights, leading to the best result. 
确定错误并收敛到不满意的结果。 VDC优于PDC，但病理曲率问题使其仅集中于一小组参数，这限制了其性能。 WPDC明确地模拟每个参数的优先级，并使用参数权重自适应地优化它们，从而获得最佳结果。

6.3. Comparison Experiments 
6.3。比较实验

In this paper, we test the performance of 3DDFA on three different tasks, including the large-pose face alignment on AFLW, 3D face alignment on AFLW2000-3D and mediumpose face alignment on 300W. 
在本文中，我们测试3DDFA在三个不同任务上的性能，包括AFLW上的大姿态面对齐，AFLW2000-3D上的3D面对齐以及300W上的中间面对齐。

6.3.1 Large Pose Face Alignment in AFLW 
6.3.1 AFLW中的大姿态面对准

Protocol: In this experiment, we regard 300W and 300WLP as the training set respectively and the whole AFLW as the testing set. The bounding boxes provided by AFLW are used for initialization (which are not the ground truth). During training, for 2D methods we use the projected 3D landmarks as the ground truth and for 3DDFA we directly regress the 3DMM parameters. During testing, we divide the testing set into 3 subsets according to their absolute yaw angles: [0◦ , 30◦ ], [30◦ , 60◦ ], and [60◦ , 90◦ ] with 11,596, 5,457 and 4,027 samples respectively. The alignment accuracy is evaluated by the Normalized Mean Error (NME), which is the average of visible landmark error normalised by the bounding box size [24, 49]. Note that the metric only considers visible landmarks and is normalized by the bounding box size instead of the common inter-pupil distance. Besides, we also report the standard deviation across testing subsets which is a good measure of pose robustness. Methods: Since little experiment has been conducted on AFLW, we choose some baseline methods with released codes, including CDM [49], RCPR [7], ESR [10] and SDM [47]. Among them ESR and SDM are popular face alignment methods in recent years. CDM is the ﬁrst one claimed to perform pose-free face alignment. RCPR is a occlusion-robust method with the potential to deal with selfocclusion and we train it with landmark visibility computed from 3D model [21]. Table. 1 demonstrates the comparison 
协议：在本实验中，我们分别将300W和300WLP作为训练集，将整个AFLW作为测试集。 AFLW提供的边界框用于初始化（这不是基本事实）。在训练期间，对于2D方法，我们使用投影的3D地标作为基础事实，对于3DDFA，我们直接回归3DMM参数。在测试过程中，我们根据它们的绝对偏航角将测试集分为3个子集：[0°，30°]，[30°，60°]和[60°，90°]分别为11,596,5,457和4,027个样本。通过归一化平均误差（NME）评估对准精度，归一化均值误差是由边界框大小[24,49]归一化的可见地标误差的平均值。请注意，度量仅考虑可见地标，并通过边界框大小而不是常见的瞳孔间距离进行归一化。此外，我们还报告了测试子集的标准偏差，这是衡量姿势稳健性的一个很好的指标。方法：由于AFLW的实验很少，我们选择一些基线方法，包括CDM [49]，RCPR [7]，ESR [10]和SDM [47]。其中ESR和SDM是近年来流行的面部对准方法。 CDM是第一个声称可以进行无姿势面部对齐的人。 RCPR是一种具有遮挡力的方法，具有处理自咬合的潜力，我们使用3D模型计算出具有里程碑意义的可视性[21]。表。图1展示了比较

Figure 9. Comparisons of cumulative errors distribution (CED) curves on AFLW. To balance the pose distribution, we plot the CED curves with a subset of 12,081 samples whose absolute yaw angles within [0◦ , 30◦ ], [30◦ , 60◦ ] and [60◦ , 90◦ ] are 1/3 each. 
图9. AFLW上累积误差分布（CED）曲线的比较。为了平衡姿态分布，我们用12,081个样本的子集绘制CED曲线，其中[0°，30°]，[30°，60°]和[60°，90°]的绝对偏航角分别为1/3 。

Results: Firstly, the results indicate that all the methods beneﬁts substantially from face proﬁling when dealing with large poses. The improvements in [60◦ , 90◦ ] are 44.06% for RCPR, 40.36% for ESR and 42.10% for SDM. This is especially impressive since the alignment models are trained on the synthesized data and tested on real samples. Thus the ﬁdelity of the face proﬁling method can be well demonstrated. Secondly, 3DDFA reaches the state of the art above all the 2D methods especially beyond medium poses. The minimum standard deviation of 3DDFA also demonstrates its robustness to pose variations. Finally, the performance of 3DDFA can be further improved with the SDM landmark reﬁnement in Section 5.2. 
结果：首先，结果表明，在处理大型姿势时，所有方法都大大受益于面部特征。 [60°，90°]的改进为RCPR为44.06％，ESR为40.36％，SDM为42.10％。这尤其令人印象深刻，因为对齐模型在合成数据上进行训练并在实际样本上进行测试。因此，可以很好地证明面部验证方法的优点。其次，3DDFA在所有2D方法之上达到了最先进的技术水平，尤其是超出中等姿势。 3DDFA的最小标准偏差也证明了其对姿势变化的稳健性。最后，通过5.2节中的SDM标志性改进，可以进一步提高3DDFA的性能。

6.3.2 
6.3.2

3D Face Alignment in AFLW2000-3D 
AFLW2000-3D中的3D面对齐

As described in Section 6.1, 3D face alignment evaluation can be degraded to all-landmark evaluation considering both visible and invisible ones. Using AFLW2000-3D as the testing set, this experiment follows the same protocol as AFLW, except 1) Instead of the visible 21 landmarks, all the MultiPIE-68 landmarks [34] in AFLW2000-3D are used for evaluation. 2) With the ground truth 3D models, the ground truth bounding boxes enclosing all the landmarks are provided for initialization. There are 1,306 samples in [0◦ , 30◦ ], 462 samples in [30◦ , 60◦ ] and 232 samples in [60◦ , 90◦ ]. The results are demonstrates in Table. 1 and the CED curves are plot in Fig. 10. We do not report the performance of provided CDM and RCPR models since they do not detect invisible landmarks. Compared with the results in AFLW, we can see the defect of barely evaluating visible landmarks. For all the methods, despite with ground 
如第6.1节所述，考虑到可见和不可见，3D面部对齐评估可以降级为全地标评估。使用AFLW2000-3D作为测试装置，该实验遵循与AFLW相同的协议，除了1）除了可见的21个界标外，AFLW2000-3D中的所有MultiPIE-68界标[34]用于评估。 2）利用地面实况3D模型，提供包围所有地标的地面真实边界框用于初始化。 [0°，30°]有1,306个样本，[30°，60°]有462个样本，[60°，90°]有232个样本。结果如表所示。图10中的CED曲线如图10所示。我们没有报告所提供的CDM和RCPR模型的性能，因为它们没有检测到不可见的地标。与AFLW的结果相比，我们可以看到几乎没有评估可见地标的缺陷。对于所有的方法，尽管有地面

152 
152

Table 1. The NME(%) of face alignment results on AFLW and AFLW2000-3D with the ﬁrst and the second best results highlighted. The bracket shows the training set. The results of provided alignment models are marked with their references. 
表1. AFLW和AFLW2000-3D的面部对齐结果的NME（％），突出显示了第一和第二个最佳结果。括号显示训练集。提供的对齐模型的结果标有其参考。

AFLW Dataset (21 pts) 
AFLW数据集（21分）

AFLW2000-3D Dataset (68 pts) 
AFLW2000-3D数据集（68分）

Method 
方法

[0, 30] 
[0,30]

[30, 60] 
[30,60]

[60, 90] Mean 
[60,90]意思是

Std 
标准

[0, 30] 
[0,30]

[30, 60] 
[30,60]

[60, 90] Mean 
[60,90]意思是

Std 
标准

CDM [49] RCPR [7] RCPR(300W) RCPR(300W-LP) ESR(300W) ESR(300W-LP) SDM(300W) SDM(300W-LP) 
CDM [49] RCPR [7] RCPR（300W）RCPR（300W-LP）ESR（300W）ESR（300W-LP）SDM（300W）SDM（300W-LP）

3DDFA 3DDFA+SDM 
3DDFA 3DDFA + SDM

8.15 6.16 5.40 5.43 5.58 5.66 4.67 4.75 
8.15 6.16 5.40 5.43 5.58 5.66 4.67 4.75

5.00 4.75 
5.00 4.75

13.02 18.67 9.80 6.58 10.62 7.12 6.78 5.55 
13.02 18.67 9.80 6.58 10.62 7.12 6.78 5.55

5.06 4.83 
5.06 4.83

16.17 34.82 20.61 11.53 20.02 11.94 16.13 9.34 
16.17 34.82 20.61 11.53 20.02 11.94 16.13 9.34

6.74 6.38 
6.74 6.38

12.44 19.88 11.94 7.85 12.07 8.24 9.19 6.55 
12.44 19.88 11.94 7.85 12.07 8.24 9.19 6.55

5.60 5.32 
5.60 5.32

4.04 14.36 7.83 3.24 7.33 3.29 6.10 2.45 
4.04 14.36 7.83 3.24 7.33 3.29 6.10 2.45

0.99 0.92 
0.99 0.92

4.16 4.26 4.38 4.60 3.56 3.67 
4.16 4.26 4.38 4.60 3.56 3.67

3.78 3.43 
3.78 3.43

9.88 5.96 10.47 6.70 7.08 4.94 
9.88 5.96 10.47 6.70 7.08 4.94

4.54 4.24 
4.54 4.24

22.58 13.18 20.31 12.67 17.48 9.76 
22.58 13.18 20.31 12.67 17.48 9.76

7.93 7.17 
7.93 7.17

12.21 7.80 11.72 7.99 9.37 6.12 
12.21 7.80 11.72 7.99 9.37 6.12

5.42 4.94 
5.42 4.94

9.43 4.74 8.04 4.19 7.23 3.21 
9.43 4.74 8.04 4.19 7.23 3.21

2.21 1.97 
2.21 1.97

Figure 10. Comparisons of cumulative errors distribution (CED) curves on AFLW2000. To balance the pose distribution, we plot the CED curves with a subset of 696 samples whose absolute yaw angles within [0◦ , 30◦ ], [30◦ , 60◦ ] and [60◦ , 90◦ ] are 1/3 each. 
图10. AFLW2000上累积误差分布（CED）曲线的比较。为了平衡姿势分布，我们用696个样本的子集绘制CED曲线，其中[0°，30°]，[30°，60°]和[60°，90°]的绝对偏航角分别为1/3 。

truth bounding boxes the performance in [60◦ , 90◦ ] and the standard deviation are obviously reduced. We think for 3D face alignment which depends on both visible and invisible landmarks [1, 55] , evaluating all the landmarks are necessary. 
真实边界框在[60°，90°]的性能和标准差明显减少。我们认为3D面部对齐取决于可见和不可见的地标[1,55]，评估所有地标是必要的。

6.3.3 Medium Pose Face Alignment 
6.3.3中间姿势面对齐

Even though not aimed at advancing face alignment in medium poses, we are also interested in the performance of 3DDFA in this popular task. The experiments are conducted on 300W following the common protocol in [54], where we use the training part of LFPW, HELEN and the whole AFW for training (3,148 images and 50,521 after augmentation), and perform testing on three parts: the test samples from LFPW and HELEN as the common subset, the 135-image IBUG as the challenging subset, and the union of them as the full set (689 images in total). The alignment accuracy are evaluated by standard landmark mean error normalised by the inter-pupil distance (NME). It can be seen in Tabel. 2 that even as a generic face alignment algorithm, 3DDFA still demonstrates competitive performance on the common 
即使不是为了在中等姿势中推进面部对齐，我们也对3DDFA在这个流行任务中的表现感兴趣。按照[54]中的常规方案在300W上进行实验，其中我们使用LFPW，HELEN和整个AFW的训练部分进行训练（增强后3,148张图像和50,521张图像），并对三个部分进行测试：测试样本来自LFPW和HELEN作为公共子集，135图像IBUG作为具有挑战性的子集，并且它们作为全集合并（总共689个图像）。通过由瞳孔间距离（NME）归一化的标准界标平均误差来评估对准精度。它可以在Tabel中看到。 2即使作为通用的面部对齐算法，3DDFA仍然表现出对常见的竞争性能

set and state-of-the-art performance on the challenging set. 
在具有挑战性的设置上设置和最先进的性能。

Table 2. The NME(%) of face alignment results on 300W, with the ﬁrst and the second best results highlighted. 
表2.面部对准的NME（％）结果为300W，突出显示第一和第二最佳结果。

Method TSPM [56] ESR [10] RCPR [7] SDM [45] LBF [32] CFSS [54] 3DDFA 3DDFA+SDM 
方法TSPM [56] ESR [10] RCPR [7] SDM [45] LBF [32] CFSS [54] 3DDFA 3DDFA + SDM

Common 8.22 5.28 6.18 5.57 4.95 4.73 6.15 5.53 
共同8.22 5.28 6.18 5.57 4.95 4.73 6.15 5.53

Challenging 18.33 17.00 17.26 15.40 11.98 9.98 10.59 9.56 
挑战18.33 17.00 17.26 15.40 11.98 9.98 10.59 9.56

Full 10.20 7.58 8.35 7.50 6.32 5.76 7.01 6.31 
满10.20 7.58 8.35 7.50 6.32 5.76 7.01 6.31

7. Conclusions 
  7.结论

In this paper, we propose a novel method, 3D Dense Face Alignment (3DDFA), which well solves the problem of face alignment across large poses. Different from traditional methods, 3DDFA skips the 2D landmark detection and starts from 3DMM ﬁtting with cascaded CNN to handle the self-occlusion problem. A face proﬁling algorithm is also proposed to synthesize face appearances in proﬁle view, providing abundant samples for training. Experiments show the state-of-the-art performance in AFLW, AFLW2000-3D and 300W. 
在本文中，我们提出了一种新的方法，3D密集面对齐（3DDFA），它很好地解决了大型姿势的面部对齐问题。与传统方法不同，3DDFA跳过2D地标检测，并从具有级联CNN的3DMM配置开始，以处理自遮挡问题。还提出了一种面部验证算法，用于在专业视图中合成面部外观，为训练提供丰富的样本。实验表明AFLW，AFLW2000-3D和300W具有最先进的性能。

8. Acknowledgment 
  8.致谢

This work was supported by the Chinese National Natural Science Foundation Projects #61375037, #61473291, #61572501, #61502491, #61572536, National Science and Technology Support Program Project #2013BAK02B01, Chinese Academy of Sciences Project No.KGZD-EW-1022, NVIDIA GPU donation program and AuthenMetric R&D Funds. 
这项工作得到了中国国家自然科学基金项目＃61375037，＃61473291，＃61572501，＃61502491，＃61572536，国家科技支撑计划项目＃2013BAK02B01，中国科学院项目编号：KGZD-EW-1022的支持， NVIDIA GPU捐赠计划和AuthenMetric研发基金。

153 
153

References 
参考

