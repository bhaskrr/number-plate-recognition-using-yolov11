# Automatic Number Plate Recognition using YOLOv11

- [Automatic Number Plate Recognition using YOLOv11](#automatic-number-plate-recognition-using-yolov11)
  - [Demo](#demo)
  - [Data](#data)
  - [Model](#model)
  - [Dependencies](#dependencies)
  - [Project Setup](#project-setup)

## Demo

<!-- ![Demo video]("./data/output/annotated_traffic.mp4") -->
<iframe width="640" height="384" src="https://github.com/bhaskrr/number-plate-recognition-using-yolov11/blob/main/data/output/annotated_traffic.mp4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Data

The **License Plate Recognition** dataset used to train this model can be found [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).

The video used in this project can be found [here](https://www.pexels.com/video/cars-are-driving-on-a-snowy-road-in-the-city-9487043/).

## Model

A license plate detector model is used to detect the license plates. The model was trained using YOLOv11 for **100** epochs with **21173** images of shape `640x640`.

The trained model is available [here](./models/license_plate_detector.pt).

## Dependencies

- Python 3.x
- opencv_contrib_python
- opencv_python
- ultralytics

## Project Setup

- Make a virtual environment using the following command:

  ```bash
  python3 -m venv myenv
  ```

  Replace `myenv` with the name you want for your virtual environment. This will create a folder named myenv in your current directory containing the virtual environment files.

- Activate the virtual environment:

  ```bash
  source myenv/bin/activate
  ```

  Remember to replace `myenv` with the actual name of the environment created in the previous step.

- Navigate to the root directory of the project:

  ```bash
  cd path/to/the/project
  ```

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- To execute the script, run:

  ```bash
  python3 main.py
  ```

- When you're done working in the virtual environment, you can deactivate it by running:
  ```bash
  deactivate
  ```
