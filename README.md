# Create embeddings for images using facenet

This small script automatically do all necessary preprocessing steps to generate
embeddings from a facenet model.

Requirements:

* python 3.6
* opencv 3.x
* tensorflow
* numpy

Facenet models need to be downloaded separately. They can be found for example
on the [Facenet github
page](https://github.com/davidsandberg/facenet#pre-trained-models).

## Facenet has an export embeddings script!

It uses quite a lot of deprecated features of old libraries. It is a rather
small script and I like to have intermediary images for validation purposes.
