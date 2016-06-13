= Preliminaries =
- for Pillow: sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
- for MatPlotLib: sudo apt-get install libfreetype6-dev libpng-dev
- for SciPy: sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose liblapack-dev gfortran

= Installation =
1. Create a virtual environment: virtualenv -p /usr/lib/python2.7 env
2. Activate virtual environment: source env/bin/activate
3. Install tensorflow (CPU only mode): 
   pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
4. git clone https://github.com/bdexp/RedCarpet-VanishingPointDetection.git
5. cd RedCarpet-VanishingPointDetection
6. Install remaining libraries: pip install -r ./RedCarpet-VanishingPointDetection/requirements.txt
7. Install python SSIM: pip install pyssim

= Installation Visualizer =
1. Create a Python 3 virtual environment: virutalenv -p /usr/lib/python3 p3env
2. Activate virtual environment: source p3env/bin/activate
3. Install tensorflow 
4. Install tensorflow (CPU only mode): pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl
5. git clone https://github.com/bdexp/RedCarpet-VanishingPointVisualizer.git
6. cd RedCarpet-VanishingPointVisualizer
7. sudo apt-get install python3.4-dev
8. Install remaining libraries: pip install -r requirements.txt
9. Change directory: cd env/lib/python3.4/site-packages
10. Create symbolic link for OpenCV 3.0: ln -s /usr/local/lib/python3.4/site-packages/cv2.cpython-34m.so cv2.so

= Execution =
$ python main.py [<operation>], with operation being one of 
    --train Trains the NN and creates a classifier in ./models/
    --preprocess Creates database of distorted and classified images
    --evaluate Evaluates test images with the trained classifier and database

The parameters for various directories can be defined in main.py at the
beginning of the function main() or in classifier/config.py.

The script run_experiments.sh will evaluate for all directories matching the 
pattern ./test-data/DS*/ and produce a ./data.csv file.

