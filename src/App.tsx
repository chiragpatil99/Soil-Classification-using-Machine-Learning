import React from 'react';
import './App.css';
import Header from './components/header/header';
import MainFeaturedPost from './components/mainfeature';
import all from './static/all.png';
import black from './static/black.jpg';
import clay from './static/clay.jpg';
import red from './static/red.jpg';
import seg_input from './static/seg_input.png';
import mask from './static/mask.png';
import segmented from './static/segmented.png';
import gb from './static/gb.png';
import feature from './static/ft.png';
import rf from './static/rf.png';
import rf_aug from './static/rf_aug.png';
import xg from './static/xg.png';
import resnet from './static/resnet.png';
import vgg from './static/vgg.png';
import xg_aug from './static/xg_aug.png';
import teaser from './static/teaser.png';
import resnet_na from './static/resnet_na.png';
import resnet_aug from './static/resnet_aug.png';
import vgg_na from './static/vgg_na.png';
import vgnet_a from './static/vgnet_a.png';
import CommonTable from './components/TableGrid';
import Typography from '@mui/material/Typography';
import ImageGrid from './components/imagegrid';
import parse from 'html-react-parser';
import { Paper } from '@mui/material';

function App() {

  const names = [{
    title: "Rutvij Wamanse - rutvijw"
  },
  {
    title: "Pranav Pawar - pranav1099"
  },
  {
    title: "Chirag Patil - chiragpatil"
  }]

  const seg_images = [
    seg_input,
    mask,
    segmented

  ];

  const dataImages = [
    all,
    black,
    red,
    clay
  ]

  const dataImagesLabels = ["Alluvial Soil", "Black Soil", "Red Soil", "Clay Soil"]
  const dataImagesLabels2 = ["Input Image", "Contoured Mask", "Segemented image"]

  const intro = {
    title: 'Introduction',
    description:
      "Soil classification is a crucial undertaking in agriculture, geology, and environmental science and holds wide-ranging implications for land use planning, crop management, and ecosystem preservation, as it is essential for comprehending and managing our planet's diverse landscapes. Conventional approaches to soil classification have frequently proven to be time-consuming and labor-intensive, necessitating extensive field surveys and expert assessments. The ability to accurately distinguish between different soil types, specifically alluvial, red, black, and clay soils, has the potential to revolutionize agricultural practices, optimizing crop selection, irrigation, and fertilization methods. By developing a robust model that can identify and classify these soil types from images, we can empower farmers, land developers, and researchers with valuable insights that can lead to more sustainable land management and greater food security. Our aim is to give a comparison between various machine learning and deep learning approaches used for the said task and analyze it for different dataset scenarios.We have used machine learning combined with image processing and feature extraction and compared its performance with deep learning techniques through convolutional neural networks (CNN) for the classification of alluvial, clay, red and black soil from images."
  };

  const abstract = {
    title: 'Abstract',
    description:
      "The aim of the project is to demonstrate the effectiveness of using computer vision techniques to manually extract features from images for classification of various types of soils. We have used image preprocessing techniques like image segmentation to extract region of interest. The feature vectors have been created by using Gabor Wavelets, Hue, Saturation, and Value  (HSV) histogram and color moments. The proposed method gives excellent results with machine learning models using these extracted features and are comparable to results obtained Deep learning based Convolutional Neural Networks for the same classification task."
  };

  const mainFeaturedPost4 = {
    title: '',
    description:
      "\n Additionally, to enhance the diversity and richness of our dataset, we'll employ augmentation techniques. Augmentation involves applying various transformations to the images, such as rotation, cropping, flipping, and adjusting brightness or contrast. These transformations not only increase the number of data points but also introduce valuable variations, which can be crucial for training robust machine learning models. Augmentation helps mitigate overfitting and provides the model with a broader range of visual contexts to learn from, ultimately improving its generalization capabilities.\n In conclusion, standardising image dimensions and using data augmentation are critical processes in preparing our dataset for analysis or training machine learning models. These measures promote consistency and enhance the dataset, allowing us to get more robust and accurate results in our efforts.\n In our analysis, we will compare and contrast two distinct techniques. The first method involves image processing, in which we extract relevant information from images. These retrieved features will be used to train our machine learning models. Our second approach, on the other hand, makes use of Convolutional Neural Networks (CNNs), a specialised deep learning architecture intended for image processing. We will analyse and compare the efficacy of these two strategies in order to determine their distinct strengths and shortcomings.\n We will use several evaluation criteria, including accuracy and the F1 score, to assess the performance of different approaches. Accuracy measures how effectively the models accurately categorise images, whereas the F1 score provides a balanced assessment of accuracy and recall, accounting for both erroneous positives and false negatives. These criteria will be critical in assessing whether strategy produces superior image categorization and analysis outcomes.\n In addition, we will run tests to see how image augmentation affects the performance of both the feature-based technique and CNN. We can examine the amount to which these alterations enhance or alter the effectiveness of the chosen approaches by comparing the results before and after augmentation.\n We hope that by conducting these rigorous comparative analyses and experiments, we will be able to provide a comprehensive understanding of the most effective approach for our specific image analysis task, taking into account both feature-based methods and deep learning with CNNs, as well as the impact of image augmentation on the final results."

  };

  const ml_approach = {
    title: 'Machine Learning Using Computer Vision Techniques',
    description: "<p>To enhance the feature extraction task from images we have used image segmentation which is dividing the image into segments of different regions and extracting the segment with the region of interest. We have used image thresholding and canny edge detection to detect the edges in an image and dilate the detected edges, further using opencv findContour function we have detected and extracted the contour with the largest area which corresponds to the soil in our case and created a zero-pixel mask for the same. Using this mask we segment the required region of interest from the image to obtain the final segmented image as the output.</p>"

  }

  const dl_approach = {
    title: 'Convolutional Neural Networks',
    description:
      "<p>The dataset is also trained using RESNET50 and VGG16 BatchNorm which are some of the well known deep convolutional networks with the help of PyTorch python library. <p/>"

  }

  const only_approach = {
    title: 'Approach',
    description: ""

  }

  const ml_approach_01 = {
    title: 'Approach',
    description:
      "Next step involved the extraction of features from the segmented soil image. In order to reduce complexity we have finalized three different types of feature that are extracted from each image namely Gabor wavelet filters , HSV (Hue, Saturation, and Value) histograms and color moments that will each provide us with unique characteristics from the soil images."

  }

  const ref = {
    title: 'References',
    description:
      "1. Barman, Utpal, and Ridip Dev Choudhury. “Soil Texture Classification Using Multi Class Support Vector Machine.” Information Processing in Agriculture 7, no. 2 (June 2020): 318–32. https://doi.org/10.1016/j.inpa.2019.08.001.\n 2. Chung, Sun-Ok, Ki-Hyun Cho, Jae-Woong Kong, Kenneth A. Sudduth, and Ki-Youl Jung. “Soil Texture Classification Algorithm Using RGB Characteristics of Soil Images.” IFAC Proceedings Volumes 43, no. 26 (2010): 34–38. https://doi.org/10.3182/20101206-3-JP-3009.00005.\n 3. He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. “Deep Residual Learning for Image Recognition,” December 10, 2015. https://arxiv.org/pdf/1512.03385.pdf.\n  4. Simonyan, Karen, and Andrew Zisserman. “Published as a Conference Paper at ICLR 2015 VERY DEEP CONVOLUTIONAL NETWORKS for LARGE-SCALE IMAGE RECOGNITION,” 2015. https://arxiv.org/pdf/1409.1556.pdf. \n 5. Chen, Tianqi, and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. 10 June 2016. \n 6. Ho, Tin Kam (1995)  Random Decision Forests. Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14–16 August 1995. pp. 278–282\n 7. Kingma, Diederik, and Jimmy Lei Ba. ADAM: A METHOD for STOCHASTIC OPTIMIZATION. 30 Jan. 2017.\n 8. Soil Image Dataset : <a href='https://www.kaggle.com/datasets/jayaprakashpondy/soil-image-dataset'>https://www.kaggle.com/datasets/jayaprakashpondy/soil-image-dataset</a>\n 9. Github Repository : <a href='https://github.com/Pranav1099/Soil-Classification-using-Machine-Learning'>https://github.com/Pranav1099/Soil-Classification-using-Machine-Learning"

  }

  const mainFeaturedPost6 = {
    title: '',
    description: "\nThe dataset we're working with comprises images of varying dimensions, which necessitates the establishment of a uniform height and width for each image. This standardization process enables us to ensure consistency and compatibility across all images, making them amenable to a comprehensive analysis."

  }

  const ml_gabor = {
    title: '',
    description:
      "<h4>Gabor Wavelet Filters<h4/> \n<p1>Gabor Wavelet filters are used to to extract distinctive features such as soil texture, structure, and patterns. These filters are specifically designed to analyze spatial frequencies in images, making them adept at capturing both fine and coarse details. The filters are characterized by their ability to represent textures at various scales and orientations, making them versatile for discerning the intricate details present in soil images. We have convolved 48 different Gabor wavelet filter kernels varying in orientation, wavelength and spatial aspect ratio using openCV with the input soil images to get the response that highlights specific textural components. This response is then used to calculate the mean-squared energy and mean amplitude as a feature vector for the said classification task. The output size of the mean-squared and mean-amplitude feature vector of the Gabor filter is 48 each which forms a combined final feature vector of size 96 per image.<p/>"

  }

  const dl_approach_1 = {
    title: '',
    description:
      "<h4>ResNet50<h4/>\n<p>ResNet50, which stands for Residual Network with 50 layers, is a deep convolutional neural network architecture that has made major advances in computer vision. ResNet50, created by Microsoft Research, is known for its exceptional depth and sophisticated design, which includes a unique residual learning method. This novel approach solves the difficulty of training very deep networks by incorporating skip links or shortcuts that bypass one or more layers, allowing information to flow smoothly during both forward and backward passes. These skip connections alleviate the vanishing gradient problem, making it possible to train incredibly deep networks with 50 layers. The deep-layer representations of the architecture contribute to its remarkable performance in image recognition tasks, making ResNet50 a popular choice for a wide range of applications, including picture categorization, object detection, and feature extraction across domains<p/>"

  }

  const dl_approach_2 = {
    title: '',
    description:
      "<h4>VGG16 BatchNorm <h4/>\n<p>VGG16 BatchNorm is an extension of the VGG16 architecture, which is a well-known deep convolutional neural network (CNN) model for image categorization. The VGG16 BatchNorm variation includes batch normalisation layers, which are an important approach in deep learning for improving model stability and efficiency. Batch normalisation normalises the input of each layer throughout the training process by modifying and scaling it, thereby minimising difficulties such as internal covariate shift. This modification to the original VGG16 architecture leads in more robust and faster training convergence. The VGG16 BatchNorm model has 16 weight layers, including convolutional and fully connected layers, and its feature maps are batch normalised. This change not only improves the model's capacity to capture complicated hierarchical information in images, but it also helps with generalisation and training speed. VGG16 BatchNorm has proven to be effective in a variety of computer vision tasks, establishing itself as an important tool in the field of deep learning for image analysis<p/>"

  }

  const ml_hsv = {
    title: '',
    description:
      "<h4>HSV Histograms<h4/>\n<p>Hue, Saturation, and Value (HSV) histograms are calculated for each segmented image. In case of soil images Hue channel can capture variations in soil color, ranging from brown and red to lighter or darker shades. Saturation is used to analyze the intensity of colors, variations in saturation may be indicative of differences in texture and moisture content of the soil. Higher saturation could suggest wetter or more textured soil, whereas Value channel captures the brightness of the soil which in turn provides insights into the distribution of brightness levels, helping to identify variations in lighting conditions across the image. The image is divided into bins within the Hue, Saturation, and Value (HSV) color space, utilizing an 8×2×2 configuration. The resulting histogram is a vector of length 32 that encapsulates information about the Hue, Saturation, and Value components of the images.<p/>"
  }

  const ml_color_m = {
    title: '',
    description:
      "<h4>Color Moments<h4/>\n<p>Color moments are statistical measures that describe the distribution of pixel intensities in an image. We have computed nine color moments for the segmented image, where three moments pertain to the mean , three for standard deviation and remaining three characterize the skewness independently for each of the three color channels in the images. The result of this approach is expressed as a feature vector with length 9.<p/>\n<p>In the final stage, the computed features are consolidated to create a single feature vector of length 137 for each image. The result of this amalgamation serves as the conclusive feature vector for the methodology. We have used this feature vector to perform multi class classification of soil images into 4 different classes namely  Alluvial Soil, Clay Soil, Red Soil and Black Soil with the help of Random Forest and XGBoost machine learning models."
  }

  const end_app = {
    title: '',
    description:
      "<p>We have performed data augmentation to enhance the robustness and diversity of our dataset. Each image has been subjected to operations such as rotation, flip and gaussian blur resulting in 3 more images.Rotation introduces diversity by changing the alignment of soil structures, capturing the inherent variability found in real-world scenarios.Horizontal flip operation simulate diverse perspectives, empowering the model to generalize effectively. Furthermore, the use of Gaussian blur enhances the dataset's robustness by introducing variations in the smoothness of images.\nBoth the approaches have been implemented using non augmented as well as augmented dataset and accuracy scores across all combinations have been analyzed and compared.<p/>"
  }

  const exp_results = {
    title: 'Experiment and Results',
    description:
      "<p>The <a href='https://www.kaggle.com/datasets/jayaprakashpondy/soil-image-dataset'>dataset</a> that we will be using consists of 4 classes namely alluvial, clay, red and black soil and total of 1563 images. We have preprocessed the data by removing duplicate images before splitting it into train and test sets to avoid data leaks from train set to test dataset in order to avoid errors in accuracy metric calculation resulting in a total of 1427 images.\nThe dataset we're working with comprises images of varying dimensions, which necessitates the establishment of a uniform height and width for each image which is chosen to be 256×256. This standardization process enables us to ensure consistency and compatibility across all images, making them amenable to a comprehensive analysis.\n The final dataset consists of a total 1440 images before augmentation and a total of 5264 images after using augmentation techinques. The ditribution of data across train and test sets for each of the four classes is as follows, <p/>"
  }

  const image_props = {
    title: '',
    description:
      "<p>Some of the sample images from each of the classess are as follows, we have considered jpg, png and jpeg formats</p>"
  }

  const ml_perf = {
    title: '',
    description:
      "<h4>Model Selection and Performance for Machine Learning Models<h4/>\n<p>GridSearchCV is a class in the scikit-learn library that performs hyperparameter tuning using an exhaustive search over a specified parameter grid. We have used 4 fold cross validation technique while performing grid search to provide a more robust and unbiased estimate of a model's performance compared to a single train-test split. The best hyperparameters are determined based on the average performance across the four folds. \n We have trained the Random Forest model using GridSearchCV for hyperparameter tuning and considered the best params for number of estimators, max_features, max_depth as 100, log2 (uses the base-2 logarithm of the total number of features for splitting a node during the construction of the decision tree) and 20 respectively for non augmented dataset. The best parameters chosen for augmented dataset were 100 estimators, sqrt option for the max_features parameter which specifies that the maximum number of features considered for splitting a node should be the square root of the total number of features and max_depth of 50.\n We also trained XGBoost model with GridSearchCV to tune with multiple hyperparatmeters and considered the best params for learning_rate, max_depth, n_estimators as 0.2, 7 and 200 respectively for non augmented dataset. The hyperparameters chosen for non augmented dataset were learning rate of 0.1, max depth of 3 and 200 estimators. The accuracy_score function from the sklearn.metrics has been used to calculate the accuracy of the models which is essentially a ratio of number of correct predictions to the total number of predictions. We finalised the mentioned two models for comparison and analysis after experimenting with a few others like KNN, PCA, SVM.\nThe test accuracies for both the models are given in the tables below.<p/>"
  }

  const dl_perf = {
    title: '',
    description:
      "<h4>Model Selection and Performance for Deep Learning Models<h4/>\n<p> In case of convolutional neural networks the images are resized to 224×224 to match the model input size. We added a linear layer at the end of these models to transform the output features to match the number of classes in our target dataset. The initial 2048-dimensional feature vector of ResNet50 was transformed into a 4-dimensional vector, whilst the 4096-dimensional feature vector of VGG16 BatchNorm received a similar transformation. We leveraged the knowledge encoded in these models from large-scale datasets using transfer learning methods. \n The training process lasted 20 epochs and used the Adam optimizer with a learning rate of 0.001 along with cross entropy as the loss function. Accuracy score is used as an evaluation metric which measures the number of correct predictions made by a model in relation to the total number of predictions made. We have calculated it by dividing the number of correct predictions by the total number of predictions. \n  Notably, both augmented and non-augmented datasets were used during training, allowing for a thorough comparison of results and emphasising the discernible impact of data augmentation on overall model performance.<p/>"
  }

  const qual_results = {
    title: 'Qualitative results',
    description: ''
  }

  const qual_results_heading = {
    title: '',
    description: '<h3>Results for Machine Learning Models<h3/>'
  }

  const qual_results_heading_2 = {
    title: '',
    description: '<h3>Results for Deep Learning Models<h3/>'
  }

  const conclusion = {
    title: 'Conclusion',
    description:
      '<p>We observed that, for the soil classification task in case of Machine Learning models Random Forest gave a better accuracy in case of both augmented and non augmented dataset as compared to XGBoost.The accuracy for these ML models were comparable to convolutional neural networks(CNN) which signifies the importance of manual feature extraction techniques specific to the type of images used. \n It appears that, particularly for smaller datasets, ML models outperform CNN models. This is because CNNs require a larger number of data points to generalize effectively and automatically learn intricate hierarchical features, whereas traditional ML models with handcrafted features using computer vision techniques proved to be more robust. Thus it can be said that in smaller datasets, manually extracted features capture relevant information more effectively than the features learned by CNN.Handcrafted features can be tailored to the specific characteristics of the dataset, providing a more focused representation. As the size of the dataset increased after augmentation the accuracy for machine learning models showed a drop on the test set whereas CNN models showed an increase in the accuracy further strenthening the argument that CNN models perform better with a large dataset size.\nHence it can be inferred that for smaller datasets it is beneficial to carefully select and engineer features that are informative for the specific task using ML models for better performance as they also often have lower cost of computation as compared to CNN models.<p/>'
  }


  const tableData = {
    data: [
      { Class: 'Alluvial Soil', TrainSet: 499, TestSet: 125 },
      { Class: 'Black Soil', TrainSet: 181, TestSet: 46 },
      { Class: 'Clay Soil', TrainSet: 214, TestSet: 54 },
      { Class: 'Red Soil', TrainSet: 246, TestSet: 62 }
    ],
    columns: ['Class', 'TrainSet', 'TestSet'],
    containerStyle: {
      backgroundColor: '#F2EAD3',
      border: '1px solid #3F2305',
      padding: '1rem 1rem',
      width: "55%",
      margin: "0rem 10rem",
      justifyContent: "center",

    },
    cellStyle: {
      backgroundColor: '#F2EAD3',
      borderBottom: '1px solid #ccc',
      verticalAlign: 'middle',
      textAlign: 'center' as const,
      fontSize: "16px",
      color: "#3F2305",

    }
  }

  const tableData_aug = {
    data: [
      { Class: 'Alluvial Soil', TrainSet: 1859, TestSet: 465 },
      { Class: 'Black Soil', TrainSet: 704, TestSet: 176 },
      { Class: 'Clay Soil', TrainSet: 793, TestSet: 199 },
      { Class: 'Red Soil', TrainSet: 854, TestSet: 214 }
    ],
    columns: ['Class', 'TrainSet', 'TestSet'],
    containerStyle: {
      backgroundColor: '#F2EAD3',
      border: '1px solid #3F2305',
      padding: '1rem 0.5rem',
      width: "55%",
      margin: "0rem 10rem",
      justifyContent: "center",

    },
    cellStyle: {
      backgroundColor: '#F2EAD3',
      borderBottom: '1px solid #ccc',
      verticalAlign: 'middle',
      textAlign: 'center' as const,
      fontSize: "16px",
      color: "#3F2305",

    }
  }

  const dl_table = {
    data: [
      { Model: 'Resnet50', Parameters: 23516228, LossFunction: 'CrossEntropy', Optmizer: 'Adam', TestAcuracy: "98.25%", Augmentation: 'No' },
      { Model: 'Vgg16bn', Parameters: 134285380, LossFunction: 'CrossEntropy', Optmizer: 'Adam', TestAcuracy: "93.72%", Augmentation: 'No' },
      { Model: 'Resnet50', Parameters: 23516228, LossFunction: 'CrossEntropy', Optmizer: 'Adam', TestAcuracy: "97.62%", Augmentation: 'Yes' },
      { Model: 'Vgg16bn', Parameters: 134285380, LossFunction: 'CrossEntropy', Optmizer: 'Adam', TestAcuracy: "96.39%", Augmentation: 'Yes' },
    ],
    columns: ['Model', 'Parameters', 'LossFunction', 'Optmizer', 'TestAcuracy', 'Augmentation'],
    containerStyle: {
      backgroundColor: '#F2EAD3',
      border: '1px solid #3F2305',
      padding: '1rem 0.5rem',
      width: "55%",
      margin: "0rem 10rem",
      justifyContent: "center",

    },
    cellStyle: {
      backgroundColor: '#F2EAD3',
      borderBottom: '1px solid #ccc',
      verticalAlign: 'middle',
      textAlign: 'center' as const,
      fontSize: "16px",
      color: "#3F2305",

    }
  }

  const ml_table = {
    data: [
      { Model: 'Random Forest', TestAcuracy: "98.25%", Augmentation: 'No' },
      { Model: 'XGBoost', TestAcuracy: "97.56%", Augmentation: 'No' },
      { Model: 'Random Forest', TestAcuracy: "98.00%", Augmentation: 'Yes' },
      { Model: 'XGBoost', TestAcuracy: " 96.58%", Augmentation: 'Yes' },
    ],
    columns: ['Model', 'TestAcuracy', 'Augmentation'],
    containerStyle: {
      backgroundColor: '#F2EAD3',
      border: '1px solid #3F2305',
      padding: '1rem 0.5rem',
      width: "55%",
      margin: "0rem 10rem",
      justifyContent: "center",

    },
    cellStyle: {
      backgroundColor: '#F2EAD3',
      borderBottom: '1px solid #ccc',
      verticalAlign: 'middle',
      textAlign: 'center' as const,
      fontSize: "16px",
      color: "#3F2305",

    }
  }

  const mainFeaturedPost4_paragraphs = mainFeaturedPost4.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const mainFeaturedPost6_paragraphs = mainFeaturedPost6.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const ml_approach_1 = ml_approach_01.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const gabor = ml_gabor.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const hsv = ml_hsv.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const color_m = ml_color_m.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const dl_app_1 = dl_approach_1.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const dl_app_2 = dl_approach_2.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));


  const img_prop = image_props.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const perf_ml = ml_perf.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const perf_dl = dl_perf.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const end = end_app.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const ml_res = qual_results_heading.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  const dl_res = qual_results_heading_2.description.split('\n').map((paragraph, index) => (
    <p key={index}>{parse(paragraph)}</p>
  ));

  return (
    <>
      <div className='container'>
        <div className="header"><Header title="Soil Type Classification using Computer Vision" sections={names} /></div>
        <MainFeaturedPost post={abstract} />
        <MainFeaturedPost post={intro} childComponent={
          <Paper elevation={3} sx={{ backgroundColor: 'transparent', boxShadow: "none", paddingLeft: "20px" }}>
            <img src={teaser} style={{ maxWidth: '100%', maxHeight: '100%', width: '85%', height: '85%', alignItems: "center" }} />
            <Typography variant="subtitle1" align="center" >
              <strong>Teaser Figure</strong>
            </Typography>
          </Paper>
        } />
        <MainFeaturedPost post={only_approach} childComponent={
          <>
            <MainFeaturedPost post={ml_approach} variantStyle={true} childComponent={
              <>
                <ImageGrid images={seg_images} labels={dataImagesLabels2} stylec={{ maxWidth: '100%', maxHeight: '100%', width: '95%', height: '80%', paddingLeft: "25px" }} />
                <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                  {ml_approach_1}{gabor}
                </Typography>
                <Paper elevation={3} sx={{ backgroundColor: 'transparent', boxShadow: "none", paddingLeft: "20px" }}>
                  <img src={gb} style={{ maxWidth: '100%', maxHeight: '100%', width: '85%', height: '85%', alignItems: "center" }} />
                  <Typography variant="subtitle1" align="center" >
                    <strong>Gabor Kernels</strong>
                  </Typography>
                </Paper>
                <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                  {hsv}
                </Typography>
                <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                  {color_m}
                </Typography>
                <Paper elevation={3} sx={{ backgroundColor: 'transparent', boxShadow: "none", paddingLeft: "20px" }}>
                  <img src={feature} style={{ maxWidth: '100%', maxHeight: '100%', width: '85%', height: '85%', alignItems: "center" }} />
                </Paper>

              </>
            } />
            <MainFeaturedPost post={dl_approach} variantStyle={true} childComponent={
              <>
                <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                  {dl_app_1}
                </Typography>
                <Paper elevation={3} sx={{ backgroundColor: 'transparent', boxShadow: "none", paddingLeft: "20px" }}>
                  <img src={resnet} style={{ maxWidth: '100%', maxHeight: '100%', width: '85%', height: '85%', alignItems: "center" }} />
                  <Typography variant="subtitle1" align="center" >
                    <strong>ResNet50</strong>
                  </Typography>
                </Paper>
                <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                  {dl_app_2}
                </Typography>
                <Paper elevation={3} sx={{ backgroundColor: 'transparent', boxShadow: "none", paddingLeft: "20px" }}>
                  <img src={vgg} style={{ maxWidth: '100%', maxHeight: '100%', width: '85%', height: '85%', alignItems: "center" }} />
                  <Typography variant="subtitle1" align="center" >
                    <strong>VGG16 BatchNorm</strong>
                  </Typography>
                </Paper>
              </>

            } />

            <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
              {end}
            </Typography></>} />
        <MainFeaturedPost post={exp_results} childComponent={

          <>
            <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", paddingLeft: "1.4rem" }}>
              <strong>Non Augmented Dataset</strong>
            </Typography>
            <div style={{ display: "flex", justifyContent: "center" }}>
              <CommonTable columns={tableData.columns} data={tableData.data} containerStyle={tableData.containerStyle} cellStyle={tableData.cellStyle} />
            </div>
            <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", paddingTop: "30px", paddingLeft: "1.4rem" }}>
              <strong>Augmented Dataset</strong>
            </Typography>
            <div style={{ display: "flex", justifyContent: "center" }}>
              <CommonTable columns={tableData_aug.columns} data={tableData_aug.data} containerStyle={tableData_aug.containerStyle} cellStyle={tableData_aug.cellStyle} />
            </div>
            <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
              {img_prop}
              <ImageGrid images={dataImages} labels={dataImagesLabels} />
            </Typography>
            <Typography style={{ paddingTop: "25px" }} className="text-body" variant="subtitle1" color="inherit" paragraph>
              {perf_ml}
              <div style={{ display: "flex", justifyContent: "center" }}>
                <CommonTable columns={ml_table.columns} data={ml_table.data} containerStyle={ml_table.containerStyle} cellStyle={ml_table.cellStyle} />
              </div>
            </Typography>
            <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
              {perf_dl}
              <div style={{ display: "flex", justifyContent: "center" }}>
                <CommonTable columns={dl_table.columns} data={dl_table.data} containerStyle={dl_table.containerStyle} cellStyle={dl_table.cellStyle} />
              </div>
            </Typography>
          </>

        } />

        <MainFeaturedPost post={qual_results} childComponent={
          <>
            <div style={{ paddingBottom: "3rem" }}>
              <Typography className="text-body" variant="subtitle1" color="inherit" paragraph>
                {ml_res}
              </Typography>
              <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", fontSize: "18px" }}>
                <strong>Random Forest</strong>
              </Typography>
              <ImageGrid images={[rf, rf_aug]} labels={["Random Forest With Augmented Dataset", "Random Forest With Non Augmented Dataset"]} stylec={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%', }} />
              <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", paddingTop: "4rem", fontSize: "18px" }}>
                <strong>XGBoost</strong>
              </Typography>
              <ImageGrid images={[xg, xg_aug]} labels={["XGBoost With Augmented Dataset", "XGBoost With Non Augmented Dataset"]} stylec={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%' }} />
            </div>
            <div style={{ paddingBottom: "3rem" }}>
              <Typography style={{ marginTop: "2rem" }} className="text-body" variant="subtitle1" color="inherit" paragraph>
                {dl_res}
              </Typography>
              <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", fontSize: "18px" }}>
                <strong>ResNet50</strong>
              </Typography>
              <ImageGrid images={[resnet_na, resnet_aug]} labels={["ResNet50 With Augmented Dataset", "ResNet50 With Non Augmented Dataset"]} stylec={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%', }} />
              <Typography component="h6" variant="h6" color="inherit" gutterBottom style={{ textAlign: "center", paddingTop: "4rem", fontSize: "18px" }}>
                <strong>VGG16 BatchNorm</strong>
              </Typography>
              <ImageGrid images={[vgg_na, vgnet_a]} labels={["VGG16 BatchNorm With Augmented Dataset", "VGG16 BatchNorm With Non Augmented Dataset"]} stylec={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%' }} />
            </div>
          </>
        } />
        <MainFeaturedPost post={conclusion} />
        <MainFeaturedPost post={ref} />
      </div>
    </>


  );
}

export default App;
