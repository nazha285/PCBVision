\# PCBVision



AI-Powered Printed Circuit Board (PCB) Defect Detection System using YOLOv8 and the DeepPCB dataset.



\## Features



\- Real-time PCB defect detection

\- Upload PCB images for inspection

\- Adjustable confidence threshold

\- Visual defect localization

\- Detection result display

\- Built with Streamlit and YOLOv8



\## Dataset



This project uses the DeepPCB dataset, which contains PCB images with annotated defects.



Defect classes include:



\- Missing Hole

\- Mouse Bite

\- Open Circuit

\- Short

\- Spur

\- Spurious Copper



\## Model



\- YOLOv8

\- Ultralytics framework



\## Web Application



Live demo:



https://pcbvision-nazha.streamlit.app



\## Technologies Used



\- Python

\- Streamlit

\- Ultralytics YOLOv8

\- OpenCV

\- NumPy

\- Pandas

\- Matplotlib



\## Installation



Clone the repository:



```bash

git clone https://github.com/nazha285/PCBVision.git

cd PCBVision

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run app.py

```



\## Project Structure



```

PCBVision/

│

├── app.py

├── predict.py

├── train.py

├── requirements.txt

├── runtime.txt

├── README.md

└── models/

```



\## Author



Nazha Al Rajab



AI + Computer Vision

