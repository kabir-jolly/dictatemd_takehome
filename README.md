# Classifying Clinical Text Take Home Project for DictateMD

This repository contains my first attempt at the take home project of classifying clinical text. Given a medical transcript, the goal was to learn which medical specialties they correspond to. The model should have generalizable performance and be able to classify unseen transcripts at test time.

## Approach 1: Baseline

The baseline approach taken was to use a Multinomial Naive Bayes model. It is the baseline as it uses pure conditional probabilities and treats the input data as a bag-of-words (the ordering of the words within the text are disregarded). I did this to see if I could set up a training pipeline as well as figure out some basic performance metrics for which I would later try and approve using a more nuanced approach.

The baseline approach was actually not bad, check the notebook for results at the bottom (especially the confusion matrix for an intuitive visualization of which classes were commonly confused, as well as under/overrepresented).

## Approach 2: Deep Learning Based

There were a couple potential models I could have tried here: RNNs, Transformers, CNNs, etc. I decided to go down the transformer route and use a pretrained BERT model and then finetune it on this medical domain. Using a pretrained BERT as a starting point is powerful as it already has a very deep understanding of language semantics/meaning. On my laptop, training a single epoch of BERT (I used the HuggingFace implementation) took over an hour so for the sake of time, I just did one. As evidenced by the results, it was so-so and was fairly good at learning the overrepresented classes. Most notably, it had a worse precision, recall, and f1-score but a higher accuracy. This is fair given that a model like this requires many more epochs and hyperparameter tuning, which was outside of my current scope.

## Limitations and Ideas for Future Work

The primary limitations include a lack of accounting for dataset limitations before training. There was a clear class imbalance (I printed out a classwise frequency count in both notebooks) where a handful of classes accounted for the vast majority of the data samples. Solutions for this would be to oversample underrepresented classes so that there is a more balanced distribution, use classwise weighting (through the loss function) to assign a higher penalty for messing up on a less frequent class, or add attention for interpretability to the BERT approach to see how we can better leverage insights from input features. Another cool idea to try out (I've never done this before but might be useful) is to use generative AI models to take in the data with small samples and generate similar datapoints to append to the dataset.

Another limitation includes the fact that we throw away a good amount of data by using pretrained BERT. The maximum input length is 512, and when looking at a distribution of transcription lengths, we clearly throw what could very well be valuable data when truncating the datapoints. Solutions for this include another layer of embedding (take the long sequences and put them in a latent embedding space) using a dimensionality reduction technique like PCA or t-SNE so that they will still be of length 512, but capture information from all parts of the longer input.

One last limitation worth mentioning is that the vectorized inputs were just created using a vanilla HuggingFace encoding utility. In past projects, I have found it to use domain-specific embedding models such as ClinicalBERT (https://github.com/kexinhuang12345/clinicalBERT) or BEHRT (https://github.com/deepmedicine/BEHRT) which would result in a far more powerful model than the generic pretrained model that was used.
