from flask import Flask, render_template, request, url_for
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = tf.keras.models.load_model('model_3.h5')












def load_and_preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_np = np.array(img) 
    return img_np, img_array

# Compute saliency map
def compute_saliency_map(model, image_array):
    if not model.optimizer:
        raise ValueError("Model must be compiled before computing saliency maps.")
    
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        pred_output = predictions[0][pred_index]
        
        grads = tape.gradient(pred_output, image_tensor)
        saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    
    saliency_map = np.clip(saliency_map, 0, np.max(saliency_map))
    saliency_map /= np.max(saliency_map)
    return saliency_map

# Apply custom colormap to saliency map
def apply_custom_colormap(saliency_map):
    colors = ["blue", "yellow", "red"]
    n_bins = 100
    cmap_name = "custom_blue_yellow_red"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    saliency_map_resized = cv2.resize(saliency_map, (saliency_map.shape[1], saliency_map.shape[0]))
    saliency_map_resized = np.uint8(255 * saliency_map_resized)
    colormap_img = custom_cmap(saliency_map_resized / 255.0)
    
    return (colormap_img[:, :, :3] * 255).astype(np.uint8)  

# Overlay saliency map on the image
def overlay_saliency_map(image, saliency_map, alpha=0.4):
    saliency_map_colored = apply_custom_colormap(saliency_map)
    overlayed_image = saliency_map_colored * alpha + image
    return np.uint8(overlayed_image)

# Save the overlayed image
def save_saliency_map(image, file_path):
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))











IMG_SIZE = (150, 150)

@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index2.html', prediction='No file part')

    imagefile = request.files['imagefile']

    if imagefile.filename == '':
        return render_template('index2.html', prediction='No selected file')

    # Save the uploaded image
    image_path = os.path.join('./static/images/', secure_filename(imagefile.filename))  # Use 'static/images/' to serve the image
    imagefile.save(image_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    prediction = model.predict(img_array)

    result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    prediction_prob = prediction[0][0]  
    
    # Calculate percentages
    pneumonia_percentage = prediction_prob * 100
    normal_percentage = (1 - prediction_prob) * 100

    original_img, processed_img = load_and_preprocess_image(image_path)
    saliency_map = compute_saliency_map(model, processed_img)
    overlayed_image = overlay_saliency_map(original_img, saliency_map)

    saliency_folder = 'static/saliency_folder'
    os.makedirs(saliency_folder, exist_ok=True)
    saliency_map_filename = os.path.basename(image_path).replace('.jpeg', '_saliency.png').replace('.jpg', '_saliency.png').replace('.png', '_saliency.png')
    saliency_map_path = os.path.join(saliency_folder, saliency_map_filename)
    save_saliency_map(overlayed_image, saliency_map_path)
    image_url1 = url_for('static', filename=f'saliency_folder/{saliency_map_filename}')

    # Send image URL along with prediction
    image_url = url_for('static', filename='images/' + secure_filename(imagefile.filename))
    
    return render_template('index2.html', prediction=result, percent1=f"{pneumonia_percentage:.2f}%", percent2=f"{normal_percentage:.2f}%", image_url=image_url,image_url1=image_url1)

if __name__ == '__main__':
    if not os.path.exists('static/images'):
        os.makedirs('static/images')  # Store images in the 'static/images' folder
    app.run(port=3000, debug=True)





