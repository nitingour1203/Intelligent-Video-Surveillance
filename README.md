# Intelligent-Video-Surveillance

Surveillance security is a very tedious and time-consuming job. In this tutorial, we will build a system to automate the task of analyzing video surveillance. We will analyze the video feed in real-time and identify any abnormal activities like violence or theft.
There is a lot of research going on in the industry about video surveillance among them; the role of CCTV videos has overgrown. CCTV cameras are placed all over the places for surveillance and security.
In the last decade, there have been advancements in deep learning algorithms for deep surveillance. These advancements have shown an essential trend in deep surveillance and promise a drastic efficiency gain. The typical applications of deep surveillance are theft identification, violence detection, and detection of the chances of explosion.


# Surveillance video analysis: relevance in present world

The main objectives identified which illustrate the relevance of the topic are listed out below.

1. Continuous monitoring of videos is difficult and tiresome for humans.

2. Intelligent surveillance video analysis is a solution to laborious human task.

3. Intelligence should be visible in all real world scenarios.

4. Maximum accuracy is needed in object identification and action recognition.

5. Tasks like crowd analysis are still needs lot of improvement.

6. Time taken for response generation is highly important in real world situation.

7. Prediction of certain movement or action or violence is highly useful in emergency situation like stampede.

8. Availability of huge data in video forms.
9. 
The majority of papers covered for this survey give importance to object recognition and action detection. Some papers are using procedures similar to a binary classification that whether action is anomalous or not anomalous. Methods for Crowd analysis and violence detection are also included. Application areas identified are included in the next section.

# Network architecture:

We have generally seen deep neural networks for computer vision, image classification, and object detection tasks. In this project, we have to extend deep neural networks to 3-dimensional for learning spatio-temporal features of the video feed.

For this video surveillance project, we will introduce a spatio temporal autoencoder, which is based on a 3D convolution network. The encoder part extracts the spatial and temporal information, and then the decoder reconstructs the frames. The abnormal events are identified by computing the reconstruction loss using Euclidean distance between original and reconstructed batch.

![image](https://github.com/nitingour1203/Intelligent-Video-Surveillance/assets/48796009/2aec3e1a-d8b6-4f5d-8598-67f3c47ed7f8)

# The dataset for abnormal event detection in video surveillance:

Following are the comprehensive datasets that are used to train models for anomaly detection tasks.

# Application areas identified

The contexts identified are listed as application areas. Major part in existing work provides solutions specifically based on the context.

1. Traffic signals and main junctions

2. Residential areas

3. Crowd pulling meetings

4. Festivals as part of religious institutions

5. Inside office buildings

Among the listed contexts crowd analysis is the most difficult part. All type of actions, behavior and movement are needed to be identified.

# Surveillance video data as Big Data

Big video data have evolved in the form of increasing number of public cameras situated towards public places. A huge amount of networked public cameras are positioned around worldwide. A heavy data stream is generated from public surveillance cameras that are creatively exploitable for capturing behaviors. Considering the huge amount of data that can be documented over time, a vital scenario is facility for data warehousing and data analysis. Only one high definition video camera can produce around 10 GB of data per day .
The space needed for storing large amount of surveillance videos for long time is difficult to allot. Instead of having data, it will be useful to have the analysis result. That will result in reduced storage space. Deep learning techniques are involved with two main components; training and learning. Both can be achieved with highest accuracy through huge amount of data.

Main advantages of training with huge amount of data are listed below. It’s possible to adapt variety in data representation and also it can be divided into training and testing equally. Various data sets available for analysis are listed below. The dataset not only includes video sequences but also frames. The analysis part mainly includes analysis of frames which were extracted from videos. So dataset including images are also useful.

The datasets widely used for various kinds of application implementation are listed in below Table 7. The list is not specific to a particular application though it is specified against an application.

# Real-time processing in video analysis
Real-time Violence Detection Framework for Football Stadiums comprising of Big Data Analysis and deep learning through Bidirectional LSTM [103] predicts violent behavior of the crowd in real-time. The real-time processing speed is achieved through SPARK frame work. The model architecture includes Apache spark framework, spark streaming, Histogram of Oriented Gradients function and bidirectional LSTM. The model takes the stream of videos from diverse sources as input. The videos are converted in the form of non-overlapping micro-batch frameworkThe spatio-temporalisspatiotemporalspatiotemporalA real-timethe  frames. Features are extracted from this group of frames through HOG FUNCTION. The images are manually modeled into different groups. The BDLSTM is trained through all these models. The SPARK framework handles the streaming data in a micro batch mode. Two kinds of processing are there like stream and batch processing.

Intelligent video surveillance for real-time detection of suicide attempts is an effort to prevent suicide by hanging in prisons. The method uses depth streams offered by an RGB-D camera. The body joints’ points are analyzed to represent suicidal behavior.
Spatio-temporal texture modeling for real-time crowd anomaly detection. Spatio-temporal texture is a combination of spatio-temporal slices and spatio-temporal volumes. The information present in these slices are abstracted through wavelet transforms. A Gaussian approximation model is applied to texture patterns to distinguish normal behaviors from abnormal behaviors.

# Deep learning models in surveillance

Deep convolutional framework for abnormal behavior detection in a smart surveillance system includes three sections.
* Human subject detection and discrimination

* A posture classification module

* An abnormal behavior detection module

The models used for above three sections are, Correspondingly

* You only look once (YOLO) network

* VGG-16 Net

* mLong short-term memory (LSTM)

For object discrimination Kalman filter based object entity discrimination algorithm is used. Posture classification study recognizes 10 types of poses. RNN uses back propagation through time (BPTT) to update weight.

The main issue identified in the method is that similar activities like pointing and punching are difficult to distinguish.

Detecting Anomalous events in videos by learning deep representations of appearance and motion proposes a new model named as AMDN. The model automatically learns feature representations. The model uses stacked de-noising auto encoders for learning appearance and motion features separately and jointly. After learning, multiple one class SVM’s are trained. These SVM predict anomaly score of each input. Later these scores are combined and detect abnormal event. A double fusion framework is used. The computational overhead in testing time is too high for real time processing.

A study of deep convolutional auto encoders for anomaly detection in videos  proposes a structure that is a mixture of auto encoders and CNN. An auto encoder includes an encoder part and decoder part. The encoder part includes convolutional and pooling layers, the decoding part include de convolutional and unpool layers. The architecture allows a combination of low level frames withs high level appearance and motion features. Anomaly scores are represented through reconstruction errors.

Going deeper with convolutions  suggests improvements over traditional neural network. Fully connected layers are replaced by sparse ones by adding sparsity into architecture. The paper suggests for dimensionality reduction which help to reduce the increasing demand for computational resources. Computing reductions happens with 1 × 1 convolutions before reaching 5 × 5 convolutions. The method is not mentioning about the execution time. Along with that not able to make conclusion about the crowd size that the method can handle successfully.

Deep learning for visual understanding: a review , reviewing the fundamental models in deep learning. Models and technique described were CNN, RBM, Autoencoder and Sparse coding. The paper also mention the drawbacks of deep learning models such as people were not able to understand the underlying theory very well.

#  crowd analysis
The review include methods which are having deep learning background and methods which are not having that background.

Spatial temporal convolutional neural networks for anomaly detection and localization in crowded scenes [114] shows the problem related with crowd analysis is challenging because of the following reasons

Large number of pedestrians

Close proximity

Volatility of individual appearance

Frequent partial occlusions

Irregular motion pattern in crowd

Dangerous activities like crowd panic

Frame level and pixel level detection

The paper suggests optical flow based solution. The CNN is having eight layers. Training is based on BVLC caffe. Random initialization of parameters is done and system is trained through stochastic gradient descent based back propagation. The implementation part is done by considering four different datasets like UCSD, UMN, Subway and finally U-turn. The details of implementation regarding UCSD includes frame level and pixel level criterion. Frame level criterion concentrates on temporal domain and pixel level criterion considers both spatiial and temporal domain. Different metrics to evaluate performance includes EER (Equal Error Rate) and Detection Rate (DR).

Online real time crowd behavior detection in video sequences [115] suggests FSCB, behavior detection through feature tracking and image segmentation. The procedure involves following steps

Feature detection and temporal filtering

Image segmentation and blob extraction

Activity detection

Activity map

Activity analysis

Alarm

The main advantage is no need of training stage for this method. The method is quantitatively analyzed through ROC curve generation. The computational speed is evaluated through frame rate. The data set considered for experiments include UMN, PETS2009, AGORASET and Rome Marathon.

Deep learning for scene independent crowd analysis [82] proposes a scene independent method which include following procedures

Crowd segmentation and detection

Crowd tracking

Crowd counting

Pedestrian travelling time estimation

Crowd attribute recognition

Crowd behavior analysis

Abnormality detection in a crowd

Attribute recognition is done thorugh a slicing CNN. By using a 2D CNN model learn appearance features then represent it as a cuboid. In the cuboid three temporal filters are identified. Then a classifier is applied on concatenated feature vector extracted from cuboid. Crowd counting and crowd density estimation is treated as a regression problem. Crowd attribute recognition is applied on WWW Crowd dataset. Evaluation metrics used are AUC and AP.

The analysis of High Density Crowds in videos [80] describes methods like data driven crowd analysis and density aware tracking. Data driven analysis learn crowd motion patterns from large collection of crowd videos through an off line manner. Learned pattern can be applied or transferred in applications. The solution includes a two step procedure. Global crowded scene matching and local crowd patch matching. Figure 2 illustrates the two step procedure.

The main advantage is no need of training stage for this method. The method is quantitatively analyzed through ROC curve generation. The computational speed is evaluated through frame rate. The data set considered for experiments include UMN, PETS2009, AGORASET and Rome Marathon.

Deep learning for scene independent crowd analysis [82] proposes a scene independent method which include following procedures

Crowd segmentation and detection

Crowd tracking

Crowd counting

Pedestrian travelling time estimation

Crowd attribute recognition

Crowd behavior analysis

Abnormality detection in a crowd

Attribute recognition is done thorugh a slicing CNN. By using a 2D CNN model learn appearance features then represent it as a cuboid. In the cuboid three temporal filters are identified. Then a classifier is applied on concatenated feature vector extracted from cuboid. Crowd counting and crowd density estimation is treated as a regression problem. Crowd attribute recognition is applied on WWW Crowd dataset. Evaluation metrics used are AUC and AP.

The analysis of High Density Crowds in videos [80] describes methods like data driven crowd analysis and density aware tracking. Data driven analysis learn crowd motion patterns from large collection of crowd videos through an off line manner. Learned pattern can be applied or transferred in applications. The solution includes a two step procedure. Global crowded scene matching and local crowd patch matching. Figure 2 illustrates the two step procedure.

![image](https://github.com/nitingour1203/Intelligent-Video-Surveillance/assets/48796009/671e71dc-ccd7-4ebe-943d-c9a5767a8522)
The database selected for experimental evaluation includes 520 unique videos with 720 × 480 resolutions. The main evaluation is to track unusual and unexpected actions of individuals in a crowd. Through experiments it is proven that data driven tracking is better than batch mode tracking. Density based person detection and tracking include steps like baseline detector, geometric filtering and tracking using density aware detector.

A review on classifying abnormal behavior in crowd scene [77] mainly demonstrates four key approaches such as Hidden Markov Model (HMM), GMM, optical flow and STT. GMM itself is enhanced with different techniques to capture abnormal behaviours. The enhanced versions of GMM are

GMM

GMM and Markov random field

Gaussian poisson mixture model and

GMM and support vector machine

GMM architecture includes components like local descriptor, global descriptor, classifiers and finally a fusion strategy. The distinction between normal and and abnormal behaviour is evaluated based on Mahalanobis distance method. GMM–MRF model mainly divided into two sections where first section identifies motion pttern through GMM and crowd context modelling is done through MRF. GPMM adds one extra feture such as count of occurrence of observed behaviour. Also EM is used for training at later stage of GPMM. GMM–SVM incorporate features such as crowd collectiveness, crowd density, crowd conflict etc. for abnormality detection.

HMM has also variants like

GM-HMM

SLT-HMM

MOHMM

HM and OSVMs

Hidden Markov Model is a density aware detection method used to detect motion based abnormality. The method generates foreground mask and perspective mask through ORB detector. GM-HMM involves four major steps. First step GMBM is used for identifying foreground pixels and further lead to development of blobs generation. In second stage PCA–HOG and motion HOG are used for feature extraction. The third stage applies k means clustering to separately cluster features generated through PCA–HOG and motion–HOG. In final stage HMM processes continuous information of moving target through the application of GM. In SLT-HMM short local trajectories are used along with HMM to achieve better localization of moving objects. MOHMM uses KLT in first phase to generate trajectories and clustering is applied on them. Second phase uses MOHMM to represent the trajectories to define usual and unusual frames. OSVM uses kernel functions to solve the nonlinearity problem by mapping high dimensional features in to a linear space by using kernel function.

In optical flow based method the enhancements made are categorized into following techniques such as HOFH, HOFME, HMOFP and MOFE.

In HOFH video frames are divided into several same size patches. Then optical flows are extracted. It is divided into eight directions. Then expectation and variance features are used to calculate optical flow between frames. HOFME descriptor is used at the final stage of abnormal behaviour detection. As the first step frame difference is calculated then extraction of optical flow pattern and finally spatio temporal description using HOFME is completed. HMOFP Extract optical flow from each frame and divided into patches. The optical flows are segmented into number of bins. Maximum amplitude flows are concatenated to form global HMOFP. MOFE method convert frames into blobs and optical flow in all the blobs are extracted. These optical flow are then clustered into different groups. In STT, crowd tracking and abnormal behaviour detection is done through combing spatial and temporal dimensions of features.

Crowd behaviour analysis from fixed and moving cameras [78] covers topics like microscopic and macroscopic crowd modeling, crowd behavior and crowd density analysis and datasets for crowd behavior analysis. Large crowds are handled through macroscopic approaches. Here agents are handled as a whole. In microscopic approaches agents are handled individually. Motion information to represent crowd can be collected through fixed and moving cameras. CNN based methods like end-to-end deep CNN, Hydra-CNN architecture, switching CNN, cascade CNN architecture, 3D CNN and spatio temporal CNN are discussed for crowd behaviour analysis. Different datasets useful specifically for crowd behaviour analysis are also described in the chapter. The metrics used are MOTA (multiple person tracker accuracy) and MOTP (multiple person tracker precision). These metrics consider multi target scenarios usually present in crowd scenes. The dataset used for experimental evaluation consists of UCSD, Violent-flows, CUHK, UCF50, Rodriguez’s, The mall and finally the worldExpo’s dataset.

Zero-shot crowd behavior recognition [79] suggests recognizers with no or little training data. The basic idea behind the approach is attribute-context cooccurrence. Prediction of behavioural attribute is done based on their relationship with known attributes. The method encompass different steps like probabilistic zero shot prediction. The method calculates the conditional probability of known to original appropriate attribute relation. The second step includes learning attribute relatedness from Text Corpora and Context learning from visual co-occurrence. Figure 3 shows the illustration of results.

![image](https://github.com/nitingour1203/Intelligent-Video-Surveillance/assets/48796009/1914fbaf-3c89-4e5f-9bb7-0983a436a516)

Computer vision based crowd disaster avoidance system: a survey [81] covers different perspectives of crowd scene analysis such as number of cameras employed and target of interest. Along with that crowd behavior analysis, people count, crowd density estimation, person re identification, crowd evacuation, and forensic analysis on crowd disaster and computations on crowd analysis. A brief summary about benchmarked datasets are also given.

Fast Face Detection in Violent Video Scenes [83] suggests an architecture with three steps such as violent scene detector, a normalization algorithm and finally a face detector. ViF descriptor along with Horn–Schunck is used for violent scene detection, used as optical flow algorithm. Normalization procedure includes gamma intensity correction, difference Gauss, Local Histogram Coincidence and Local Normal Distribution. Face detection involve mainly two stages. First stage is segmenting regions of skin and the second stage check each component of face.

Rejecting Motion Outliers for Efficient Crowd Anomaly Detection [54] provides a solution which consists of two phases. Feature extraction and anomaly classification. Feature extraction is based on flow. Different steps involved in the pipeline are input video is divided into frames, frames are divided into super pixels, extracting histogram for each super pixel, aggregating histograms spatially and finally concatenation of combined histograms from consecutive frames for taking out final feature. Anomaly can be detected through existing classification algorithms. The implementation is done through UCSD dataset. Two subsets with resolution 158 × 238 and 240 × 360 are present. The normal behavior was used to train k means and KUGDA. The normal and abnormal behavior is used to train linear SVM. The hardware part includes Artix 7 xc7a200t FPGA from Xilinx, Xilinx IST and XPower Analyzer.

Deep Metric Learning for Crowdedness Regression [84] includes deep network model where learning of features and distance measurements are done concurrently. Metric learning is used to study a fine distance measurement. The proposed model is implemented through Tensorflow package. Rectified linear unit is used as an activation function. The training method applied is gradient descent. Performance is evaluated through mean squared error and mean absolute error. The WorldExpo dataset and the Shanghai Tech dataset are used for experimental evaluation.

A Deep Spatiotemporal Perspective for Understanding Crowd Behavior [61] is a combination of convolution layer and long short-term memory. Spatial informations are captured through convolution layer and temporal motion dynamics are confined through LSTM. The method forecasts the pedestrian path, estimate the destination and finally categorize the behavior of individuals according to motion pattern. Path forecasting technique includes two stacked ConvLSTM layers by 128 hidden states. Kernel of ConvLSTM size is 3 × 3, with a stride of 1 and zeropadding. Model takes up a single convolution layer with a 1 × 1 kernel size. Crowd behavior classification is achieved through a combination of three layers namely an average spatial pooling layer, a fully connected layer and a softmax layer.

Crowded Scene Understanding by Deeply Learned Volumetric Slices [85] suggests a deep model and different fusion approaches. The architecture involves convolution layers, global sum pooling layer and fully connected layers. Slice fusion and weight sharing schemes are required by the architecture. A new multitask learning deep model is projected to equally study motion features and appearance features and successfully join them. A new concept of crowd motion channels are designed as input to the model. The motion channel analyzes the temporal progress of contents in crowd videos. The motion channels are stirred by temporal slices that clearly demonstrate the temporal growth of contents in crowd videos. In addition, we also conduct wide-ranging evaluations by multiple deep structures with various data fusion and weights sharing schemes to find out temporal features. The network is configured with convlutional layer, pooling layer and fully connected layer with activation functions such as rectified linear unit and sigmoid function. Three different kinds of slice fusion techniques are applied to measure the efficiency of proposed input channels.

Crowd Scene Understanding from Video A survey [86] mainly deals with crowd counting. Different approaches for crowd counting are categorized into six. Pixel level analysis, texture level analysis, object level analysis, line counting, density mapping and joint detection and counting. Edge features are analyzed through pixel level analysis. Image patches are analysed through texture level analysis. Object level analysis is more accurate compared to pixel and texture analysis. The method identifies individual subjects in a scene. Line counting is used to take the count of people crossed a particular line.

![image](https://github.com/nitingour1203/Intelligent-Video-Surveillance/assets/48796009/67a57327-690f-4594-9f9d-e1d04fee310f)

![image](https://github.com/nitingour1203/Intelligent-Video-Surveillance/assets/48796009/66cb9612-2c85-496f-8854-776e38aa80df)


As an analysis of existing methods the following shortcomings were identified. Real world problems are having following objectives like

Time complexity

Bad weather conditions

Real world dynamics

Occulsions

Overlapping of objects

Existing methods were handling the problems separately. No method handles all the objectives as features in a single proposal.

To handle effective intelligent crowd video analysis in real time the method should be able to provide solutions to all these problems. Traditional methods are not able to generate efficient economic solution in a time bounded manner.

The availability of high performance computational resource like GPU allows implementation of deep learning based solutions for fast processing of big data. Existing deep learning architectures or models can be combined by including good features and removing unwanted features.

# Conclusion
The paper reviews intelligent surveillance video analysis techniques. Reviewed papers cover wide variety of applications. The techniques, tools and dataset identified were listed in form of tables. Survey begins with video surveillance analysis in general perspective, and then finally moves towards crowd analysis. Crowd analysis is difficult in such a way that crowd size is large and dynamic in real world scenarios. Identifying each entity and their behavior is a difficult task. Methods analyzing crowd behavior were discussed. The issues identified in existing methods were listed as future directions to provide efficient solution.

# Abbreviations SVAS:
Surveillance Video Analysis System

IBSTM: Interval-Based Spatio-Temporal Model

KLT: Kanade–Lucas–Tomasi

GMM: Gaussian Mixture Model

SVM: Support Vector Machine

DAAL: Deep activation-based attribute learning

HMM: Hidden Markov Model

YOLO: You only look once

LSTM: Long short-term memory

AUC: Area under the curve

ViF: Violent flow descriptor
