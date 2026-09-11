# Dataset layout

Create these four folders and put photographs in each:

```text
data/
  plastic/
  general/  # or paper/, but not both
  metal/
  other/
```

The easiest collector is `python dataset_studio.py --middle general`. You can also use `python collect_images.py plastic` and repeat for each class. Collect at least 300 varied images per class for a first experiment; 800–1500 per class is much healthier. Photograph multiple object types, rotations, distances, backgrounds, hands and lighting conditions. Do not split frames from one short video across training and validation because nearly identical frames exaggerate accuracy.

The `other` class is essential. Include food, glass, electronics, hands, empty inspection pads, multiple objects and anything that must not open a lid.
