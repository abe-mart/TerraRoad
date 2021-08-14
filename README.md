# TerraRoad

TerraRoad is a Terragen road creator!  It combines a heightfield exported from Terragen with an svg path defining a road shape, and creates a new heightfield with the road path leveled and smoothed.  The tool also creates a variety of masks that can be used in Terragen for shading the road and shoulder.  

![Road picture in Terragen](./Images/RoadSampleSmall.jpg)

## Installation

To use TerraRoad, you'll need Python installed on your computer.  On Windows, Python can be obtained easily through the [Microsoft Store](https://www.microsoft.com/en-us/p/python-39/9p7qfqmjrfp7).  For installation on Mac or Linux, see [python.org](https://www.python.org/downloads/).

Once Python is installed, open the command prompt and use the Python package manager [pip](https://pip.pypa.io/en/stable/) to install the packages needed to run TerraRoad.

```bash
pip install pysimplegui imageio numpy scipy scikit-image scikit-learn svgpathtools
```

Now you are ready to download TerraRoad.  TerraRoad is contained in a single Python script, TerraRoad.py, which you can download by clicking the green 'Code' button above, and then 'Download ZIP'.  Unzip the folder, and navigate to the unzipped folder in your command prompt.  Launch TerraRoad by running:

```
python TerraRoad.py
```

![TerraRoad UI](./Images/UI.jpg)

## Usage

## Example Files

![Heightfield Node](./Images/Tutorial1.jpg)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
