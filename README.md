# mtgir

## Requirements

- python 3.10+ - <https://www.python.org/downloads/>
- `poetry` - <https://github.com/python-poetry/poetry>
- `jq` - <https://stedolan.github.io/jq/>
- Nice to have: `imgcat` - <https://github.com/eddieantonio/imgcat>

## Getting Started

```shell
poetry install
```

### Download all images from the latest dataset

```shell
poetry run mtgir --download
```

### Find the best match for a given image

> Note: You must have download and loaded a dataset first.

```shell
poetry run mtgir /path/to/test/image.jpg
```

#### Example

![example](https://raw.githubusercontent.com/lexicalunit/mtgir/main/example.png)

## Local Tests

> Note: You must have download and loaded a dataset first.

```shell
poetry run pytest
```

## Research

- [OpenCV Python Feature Detection Cheatsheet](https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md)
- [ORB (Oriented FAST and Rotated BRIEF) OpenCV Tutorial](https://docs.opencv.org/3.4.4/d1/d89/tutorial_py_orb.html)
- [Robust image matching via ORB feature and VFC for mismatch removal](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10609/2283260/Robust-image-matching-via-ORB-feature-and-VFC-for-mismatch/10.1117/12.2283260.short)
- [MTG-Card-Reader-Web Identification](https://github.com/TrifectaIII/MTG-Card-Reader-Web/blob/01fcbcdddf2f0bc019010062c1ec35b21cd9b9c4/Identification.py)
- [OpenCV / Python : Fast way to match a picture with a database](https://stackoverflow.com/questions/29563429/opencv-python-fast-way-to-match-a-picture-with-a-database)
- [OpenCV's Flann matcher index in python](https://stackoverflow.com/questions/54208099/opencvs-flann-matcher-index-in-python)
- [Find image from a database of images](https://stackoverflow.com/questions/46479237/find-image-from-a-database-of-images)
- [[D] What would be the best way to match an image to a database of existing images?](https://www.reddit.com/r/MachineLearning/comments/xc0tmt/d_what_would_be_the_best_way_to_match_an_image_to/)
- [How to I compute matching features between high resolution images?](https://stackoverflow.com/questions/64246583/how-to-i-compute-matching-features-between-high-resolution-images)
- [Image Similarity Search in PyTorch
](https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469)
- [Exploring Data Augmentation with Keras and TensorFlow](https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844)
