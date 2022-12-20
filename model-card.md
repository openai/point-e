# Model Card: Point-E

This is the official codebase for running the point cloud diffusion models and SDF regression models described in [Point-E: A System for Generating 3D Point Clouds from Complex Prompts](https://arxiv.org/abs/2212.08751). These models were trained and released by OpenAI.
Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993), we're providing some information about how the models were trained and evaluated. 

# Model Details

The Point-E models are trained for use as point cloud diffusion models and SDF regression models.
Our image-conditional models are often capable of producing coherent 3D point clouds, given a single rendering of a 3D object. However, the models sometimes fail to do so, either producing incorrect geometry where the rendering is occluded, or producing geometry that is inconsistent with visible parts of the rendering. The resulting point clouds are relatively low-resolution, and are often noisy and contain defects such as outliers or cracks.
Our text-conditional model is sometimes capable of producing 3D point clouds which can be recognized as the provided text description, especially when the text description is simple. However, we find that this model fails to generalize to complex prompts or unusual objects.

## Model Date

December 2022

## Model Versions

 * `base40M-imagevec` - a 40 million parameter image to point cloud model that conditions on a single CLIP ViT-L/14 image vector. This model can be used to generate point clouds from rendered images, but does not perform as well as our other models for this task.
 * `base40M-textvec` - a 40 million parameter text to point cloud model that conditions on a single CLIP ViT-L/14 text vector. This model can be used to directly generate point clouds from text descriptions, but only works for simple prompts.
 * `base40M-uncond` - a 40 million parameter point cloud diffusion model that generates unconditional samples. This is included only as a baseline.
 * `base40M` - a 40 million parameter image to point cloud diffusion model that conditions on the latent grid from a CLIP ViT-L/14 model. This model can be used to generate point clouds from rendered images, but is not as good as the larger models trained on the same task.
 * `base300M` - a 300 million parameter image to point cloud diffusion model that conditions on the latent grid from a CLIP ViT-L/14 model. This model can be used to generate point clouds from rendered images, but it is slightly worse than base1B
 * `base1B` - a 1 billion parameter image to point cloud diffusion model that conditions on the latent grid from a CLIP ViT-L/14 model.
 * `upsample` - a 40 million parameter point cloud upsampling model that can optionally condition on an image as well. This takes a point cloud of 1024 points and upsamples it to 4096 points.
 * `sdf` - a small model for predicting signed distance functions from 3D point clouds. This can be used to predict meshes from point clouds.
 * `pointnet` - a small point cloud classification model used for our P-FID and P-IS evaluation metrics.

## Paper & samples

[Paper](https://arxiv.org/abs/2212.08751) / [Sample point clouds](point_e/examples/paper_banner.gif)

# Training data

These models were trained on a dataset of several million 3D models. We filtered the dataset to avoid flat objects, and used [CLIP](https://github.com/openai/CLIP/blob/main/model-card.md) to cluster the dataset and downweight clusters of 3D models which appeared to contain mostly unrecognizable objects. We additionally down-weighted clusters which appeared to consist of many similar-looking objects. We processed the resulting dataset into renders (RGB point clouds of 4K points each) and text captions from the associated metadata.
Our SDF regression model was trained on a subset of the above dataset. In particular, we only retained 3D meshes which were manifold (i.e. watertight and free of singularities).

# Evaluated Use

We release these models to help advance research in generative modeling. Due to the limitations and biases of our models, we do not currently recommend it for commercial use. We understand that our models may be used in ways we haven't anticipated, and that it is difficult to define clear boundaries around what constitutes appropriate "research" use. In particular, we caution against using these models in applications where precision is critical, as subtle flaws in the outputs could lead to errors or inaccuracies.
Functionally, these models are trained to be able to perform the following tasks for research purposes, and are evaluated on these tasks:

 * Generate 3D point clouds conditioned on single rendered images
 * Generate 3D point clouds conditioned on text
 * Create 3D meshes from noisy 3D point clouds

Our image-conditional models are intended to produce coherent point clouds, given a representative rendering of a 3D object. However, at their current level of capabilities, the models sometimes fail to generate coherent output, either producing incorrect geometry where the rendering is occluded, or producing geometry that is inconsistent with visible parts of the rendering. The resulting point clouds are relatively low-resolution, and are often noisy and contain defects such as outliers or cracks. 

Our text-conditional model is sometimes capable of producing 3D point clouds which can be recognized as the provided text description, especially when the text description is simple. However, we find that this model fails to generalize to complex prompts or unusual objects.

# Performance and Limitations

Our image-conditional models are limited by the text-to-image model that is used to produce synthetic views. If the text-to-image model contains a bias or fails to understand a particular concept, these limitations will be passed down to the image-conditional point cloud model through conditioning images.
While our main focus is on image-conditional models, we also experimented with a text-conditional model. We find that this model can sometimes produce 3D models of people that exhibit gender biases (for example, samples for "a man" tend to be wider and less narrow than samples for "a woman"). We additionally find that this model is sometimes capable of producing violent objects such as guns or tanks, although these generations are always low-quality and unrealistic.

Since our dataset contains many simplistic, cartoonish 3D objects, our models are prone to mimicking this style.

While these models were developed for research purposes, they have potential implications if used more broadly. For example, the ability to generate 3D point clouds from single images could help advance research in computer graphics, virtual reality, and robotics. The text-conditional model could allow for users to easily create 3D models from simple descriptions, which could be useful for rapid prototyping or 3D printing.
 
The combination of these models with 3D printing could potentially be harmful, for example if used to prototype dangerous objects or when parts created by the model are trusted without external validation.

Finally, point cloud models inherit many of the same risks and limitations as image-generation models, including the propensity to produce biased or otherwise harmful content or to carry dual-use risk. More research is needed on how these risks manifest themselves as capabilities improve.

