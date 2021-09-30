import io
import os
from base64 import b64encode
from IPython.display import HTML
!git clone https: // github.com / theAIGuysCode / yolov4 - deepsort

!pip install - r requirements-gpu.txt
# download yolov4 model weights to data folder
!wget https: // github.com / AlexeyAB / darknet / releases / download / darknet_yolo_v3_optimal / yolov4.weights - P data /


!python save_model.py - -model yolov4

# define helper function to display videos


def show_video(file_name, width=640):
    # show resulting deepsort video
    mp4 = open(file_name, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
  <video width="{0}" controls>
        <source src="{1}" type="video/mp4">
  </video>
  """.format(width, data_url))


    # convert resulting video from avi to mp4 file format
path_video = os.path.join("outputs", "tracker.avi")
%cd outputs/
!ffmpeg - y - loglevel panic - i tracker.avi output.mp4
%cd ..

# output object tracking video
path_output = os.path.join("outputs", "output.mp4")
show_video(path_output, width=960)


# convert resulting video from avi to mp4 file format
path_video = os.path.join("outputs", "custom.avi")
%cd outputs/
!ffmpeg - y - loglevel panic - i custom.avi result.mp4
%cd ..

# output object tracking video
path_output = os.path.join("outputs", "result.mp4")
show_video(path_output, width=960)
