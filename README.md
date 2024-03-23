# RenAIssance-Generator

## Problem Description

In recent years, there has been a dedicated effort in the scientific community to generate creative products such as poetry, stories, music, and art using artificial intelligence. One such project by Elgammal et al. introduced a Generative Adversarial Network (GAN) system capable of generating art by learning about styles and deviating from style norms. Specifically, this project focuses on the Renaissance Oil Painting style. However, existing AI systems that generate Renaissance-style paintings suffer from limitations, including biases inherent in the training data.

This project aims to address these limitations by utilizing a GAN-based artificial intelligence approach to generate images with a Renaissance touch. The method involves training a GAN to learn and replicate the style of Renaissance paintings while avoiding biases present in the original dataset.

## Method

The project utilizes a Generative Adversarial Network (GAN) consisting of two sub-models: the Generator and the Discriminator. The Generator generates images with a Renaissance style, while the Discriminator evaluates the authenticity of the generated images. The GAN is trained using a dataset of Renaissance paintings, adjusting parameters such as batch size and number of epochs to optimize training.

The architecture is implemented using TensorFlow and Keras, with the Generator comprising 22 layers and the Discriminator comprising 14 layers. Training involves preprocessing the input images, training the Discriminator to distinguish between real and fake images, and updating the weights of both models using backpropagation. Visually the training looks like this:

![image](https://github.com/FierceWeirdo/RenAIssance-Generator/assets/77613181/56e522e8-c208-45c9-9921-7964147f4045)

And the iterations as follows:

![image](https://github.com/FierceWeirdo/RenAIssance-Generator/assets/77613181/09498f5e-bf38-4957-895d-6cef399f036c)


## Materials

### Hardware:
- Hewlett-Packard 64-bit laptop with an 11th Gen Intel(R) Core(TM) i5-1135G7 processor and 16.0 GB RAM

### Software:
- Python with TensorFlow, Keras, numpy, and cv2 libraries

### Dataset:
- A subset of the ArtDL dataset containing 7000 Renaissance paintings

### Additional Materials:
- "Generative Adversarial Networks with Python" by Jason Brownlee
- Online resources from the Python and TensorFlow websites

## Evaluation

The project underwent two types of evaluation:

1. **Preservation of Skin Tone:** Skin tone preservation was evaluated using a Skin Color Similarity Index (SCSI) algorithm, ensuring that the generated images retained the skin tones of the original images.

2. **Visual Quality and Style Consistency:** Human raters evaluated the visual quality and style consistency of the generated images. While the generated landscape paintings performed relatively well, the portraits scored lower in terms of resemblance to human-made art and Renaissance authenticity.

   Examples from surveys:

   ![image](https://github.com/FierceWeirdo/RenAIssance-Generator/assets/77613181/4538b08e-e270-4fa7-a6e2-1d87706a9577)


## Limitations

- Limited availability of training data and hardware constraints led to compromises in dataset size and image resolution, affecting the quality of generated images.
- Bias in the dataset towards light-skinned portraits posed challenges in preserving skin tone diversity.
- Hardware limitations restricted the size of the training dataset, impacting the quality of generated images.

## Next Steps

1. Experiment with GAN parameters and explore the effects on output quality.
2. Train the GAN on a larger, more diverse dataset to improve output quality.
3. Explore different artistic styles and train the GAN on those images.
4. Develop a user interface for transforming user-uploaded images into Renaissance paintings.
5. Investigate implementing Neural Style Transfer using the GAN.

Efforts will focus on optimizing the GAN's efficiency within the current hardware constraints to enhance output quality. Further experimentation and refinement are essential for achieving more realistic and diverse Renaissance-style paintings.
