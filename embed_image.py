import pathlib
import functools
import argparse

import numpy as np

import cv2

import tensorflow as tf
from facenet.src.align import detect_face
from facenet.src import facenet


def load_image(path):
    img = cv2.imread(str(path))
    # switch to rgb, since cv2 defaults to bgr
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def imload(fun):
    @functools.wraps(fun)
    def inner(self, img, *args, **kwargs):
        if isinstance(img, str) or isinstance(img, pathlib.Path):
            img = load_image(img)
        return fun(self, img, *args, **kwargs)
    return inner


class FaceAligner:

    def __init__(
            self,
            modelpath="facenet/src/align",
            minsize=20,
            threshold=(0.6, 0.7, 0.7),
            factor=0.709):
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor
        self.sess = tf.Session()
        self.funs = detect_face.create_mtcnn(self.sess, "facenet/src/align")

    @imload
    def detect(self, image):
        """Detect a face using the given image."""
        total_boxes, _ = detect_face.detect_face(
            image,
            self.minsize, *self.funs, threshold=self.threshold, factor=self.factor)
        return total_boxes

    @imload
    def crop(self, image, margin=22):
        """Crop faces with a given margin.
        """
        total_boxes = self.detect(image)
        cropped = [
            image[
                int(np.maximum(box[1] - margin, 0)): int(np.minimum(box[3] + margin, image.shape[0])),
                int(np.maximum(box[0] - margin, 0)): int(np.minimum(box[2] + margin, image.shape[1])),
                :
            ]
            for box in total_boxes if box[4] > 0.5
        ]
        return cropped

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.sess.close()


class FacenetEmbedding:
    def __init__(self, modelpath, image_size=160):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            facenet.load_model(modelpath)

        self._input = self.graph.get_tensor_by_name("input:0")
        self._embeddings = self.graph.get_tensor_by_name("embeddings:0")
        self._phase_train = self.graph.get_tensor_by_name("phase_train:0")

        # define input and output sizes
        self.image_size = image_size
        self.embedding_size = self._embeddings.get_shape()[1]

    def __enter__(self):
        return self

    def get_embedding(self, img):
        img = img.reshape(-1, self.image_size, self.image_size, 3)
        emb = self.sess.run(
            self._embeddings, {self._input: img, self._phase_train: False},
        )
        print(emb.shape)
        return emb

    def __exit__(self, *_):
        self.sess.close()


def ensure_outdir(fun):
    @functools.wraps(fun)
    def inner(inpath, outpath, *args, **kwargs):
        outpath.mkdir(parents=True, exist_ok=True)
        return fun(inpath, outpath, *args, **kwargs)
    return inner


@ensure_outdir
def align_faces(inpath, output, model_align="facenet/src/align"):
    with FaceAligner(model_align) as fa:
        for image in faces.glob("*"):
            if image.is_dir():
                continue
            crops = fa.crop(image)
            for i, crop in enumerate(crops):
                outpath = output / f"{image.stem}_{i}.png"
                save_image(outpath, crop)


@ensure_outdir
def transform_images(inpath, output, image_size=160):
    for imgpath in inpath.glob("*0.png"):
        img = load_image(imgpath)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        img = facenet.prewhiten(img)
        outpath = output / f"{imgpath.stem}.npy"
        np.save(outpath, img)


@ensure_outdir
def embed_images(inpath, output, model):
    fn = FacenetEmbedding(model)
    for nppath in inpath.glob("*.npy"):
        indata = np.load(nppath)
        emb = fn.get_embedding(indata)
        outpath = output / f"{nppath.stem}.npy"
        np.save(outpath, emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create embeddings from facial images. Will handle all necessary preprocessing steps.")
    parser.add_argument(
        "input", help="Input directory containing images",
        default="data/faces", type=pathlib.Path,
    )
    parser.add_argument(
        "--output", help="Output directory containing intermediary steps and embeddings",
        default="output", type=pathlib.Path,
    )
    parser.add_argument(
        "--model", help="Path to facenet model.", required=True,
    )
    parser.add_argument(
        "--model-align", help="Path to face alignment model",
        default="facenet/src/align", type=pathlib.Path)
    parser.add_argument(
        "--force", help="Force recreation of intermediary data.", action="store_true")
    args = parser.parse_args()
    faces = args.input

    aligned = args.output / "aligned"
    if not aligned.exists() or args.force:
        align_faces(faces, aligned, args.model_align)
    preprocessed = args.output / "scaled"
    if not preprocessed.exists() or args.force:
        transform_images(aligned, preprocessed)
    embeddings = args.output / "embeddings"
    if not embeddings.exists() or args.force:
        embed_images(preprocessed, embeddings, args.model)

    print(f"Embeddings can be found in {embeddings}")
