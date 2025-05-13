
---

# 🎥 Real-Time Crowd Analysis using YOLOv3 & OpenCV

A computer vision-based system that detects, counts, and analyzes people in video frames in real time. It provides alerts when crowd size or density exceeds safe limits, making it useful for public safety, smart surveillance, and event monitoring.

---

## 📌 Features

* ✅ Real-time **person detection** using YOLOv3
* ✅ **People counting** in each video frame
* ✅ **Space occupancy analysis** (density estimation)
* ✅ Customizable **crowd and density thresholds**
* ✅ Visual alerts directly on video feed
* ✅ Scalable to work with video files or live webcam

---

## 🛠️ Technologies Used

* **Python 3**
* **OpenCV**
* **YOLOv3 (Darknet)**
* **NumPy**
* **Tkinter** (for adaptive display)

---

## 🚀 How It Works

1. YOLOv3 detects all persons in the frame.
2. For each detection, it calculates bounding boxes.
3. Total area occupied by people is compared against the frame area.
4. Alerts are triggered if:

   * People count > set threshold
   * Space occupancy ratio > defined limit

---

## 📂 Project Structure

```bash
├── main.py                 # Main Python script
├── yolov3.cfg              # YOLOv3 configuration file
├── yolov3.weights          # Pre-trained weights file
├── coco.names              # Class labels (COCO dataset)
```

> Note: Make sure to place all files in the same directory or update the paths accordingly.

---

## 🖥️ Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/real-time-crowd-analysis.git
   cd real-time-crowd-analysis
   ```

2. Download YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights) and place them in the project folder.

3. Install dependencies:

   ```bash
   pip install opencv-python numpy
   ```

4. Run the script:

   ```bash
   python main.py
   ```

> You can modify the `video_path` in `main.py` to use a different video file or live webcam (`0`).

---

## 📸 Sample Output


![image](https://github.com/user-attachments/assets/b167ca66-d0b9-418a-917f-0479c6d4641c)


---

## 💡 Future Improvements

* Add real-time anomaly detection
* Integrate with Streamlit for web-based UI
* Deploy on edge devices like Raspberry Pi
* Save crowd stats for analytics

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue or suggest a feature.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🔗 Connect With Me

Feel free to connect with me on [LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/harsh-gupta-22a24a286?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)) or check out more of my projects.

---


