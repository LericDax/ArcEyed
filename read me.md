ArcEyed

© 2023 LAK, Leric Dax, and Azoth Corp.
https://github.com/LericDax/ArcEyed
www.LericDax.com


ArcEyed is a Python application that detects and matches eyebrows in images. 
The application uses dlib, OpenCV, and PyQt5 to create a graphical user interface for selecting images and matching eyebrows.




Installation

Clone the repository or download the source code.

Create a virtual environment and activate it.

Install the required packages using the following command:

	pip install -r requirements.txt

Download the required dlib model files:

	Download shape_predictor_68_face_landmarks.dat from here (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). After downloading, extract the .bz2 file to obtain the .dat file.
	Download dlib_face_recognition_resnet_model_v1.dat from here (http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2). After downloading, extract the .bz2 file to obtain the .dat file.

Place the downloaded .dat files in the same directory as the main.py file.





Usage

Run the application using the following command from the ArcEyed folder:

	python main.py

Browse and select a source image file (JPEG, JPG, or PNG) containing the eyebrows you want to match.

Browse and select an image folder containing the images you want to search for matching eyebrows.

Browse and select an output folder where the matched images will be saved.

Click on the "Match Eyebrows" button to start the process. The matched images will be displayed in the "Matched Images" section of the user interface and saved in the specified output folder.







Dependencies
dlib
numpy
opencv-python
torch
PyQt5
openai
clip






ArcEyed

MIT License

Copyright ©  2023 LAK, Leric Dax, and Azoth Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


https://github.com/LericDax/ArcEyed
www.LericDax.com
