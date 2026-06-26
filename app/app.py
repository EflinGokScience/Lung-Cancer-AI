import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

MODEL_PATH = "model/efficientnet_gradcam_model.keras"

model = load_model(MODEL_PATH, compile=False)

class_names = ["adenocarcinoma", "large_cell", "normal", "squamous"]

base_model = model.get_layer("efficientnetb0")
last_conv_layer = base_model.get_layer("top_conv")

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, base_model.output]
)

base_index = None
for i, layer in enumerate(model.layers):
    if layer.name == "efficientnetb0":
        base_index = i
        break

classifier_layers = model.layers[base_index + 1:]


def create_probability_chart(preds):
    display_names = [
        "Adenocarcinoma",
        "Large cell",
        "Normal",
        "Squamous"
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(display_names, preds * 100)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Class Probability Scores")
    ax.tick_params(axis="x", rotation=20)

    for i, value in enumerate(preds * 100):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.canvas.draw()
    chart = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return chart


def predict_ct(uploaded_img):
    if uploaded_img is None:
        return (
            "Please upload a CT scan image first.",
            None,
            None,
            None,
            None,
            "Waiting for image upload."
        )

    img = uploaded_img.convert("RGB").resize((224, 224))
    original = np.array(img)

    input_array = np.expand_dims(original.astype(np.float32), axis=0)

    processed = tf.keras.applications.efficientnet.preprocess_input(
        input_array.copy()
    )

    preds = model.predict(processed, verbose=0)[0]

    pred_index = np.argmax(preds)
    pred_label = class_names[pred_index]
    confidence = preds[pred_index] * 100

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_extractor(processed)

        x = base_outputs
        for layer in classifier_layers:
            x = layer(x, training=False)

        loss = x[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    probability_chart = create_probability_chart(preds)

    display_label = pred_label.replace("_", " ").title()

    result_text = f"""
## Prediction: **{display_label}**

### Confidence: **{confidence:.2f}%**

### Class Probabilities
- Adenocarcinoma: **{preds[0] * 100:.2f}%**
- Large cell carcinoma: **{preds[1] * 100:.2f}%**
- Normal: **{preds[2] * 100:.2f}%**
- Squamous cell carcinoma: **{preds[3] * 100:.2f}%**
"""

    explanation_text = f"""
### AI Explanation

The model classified the uploaded CT image as **{display_label}** with **{confidence:.2f}% confidence**.

The Grad-CAM heatmap highlights image regions that most influenced the model's decision.  
Warmer areas indicate regions the AI considered more important during classification.

**Important:** This tool is for research and educational purposes only. It is **not** intended for clinical diagnosis.
"""

    return (
        result_text,
        original,
        heatmap_color,
        overlay,
        probability_chart,
        explanation_text
    )


custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #06111f, #0b1f35);
}

h1, h2, h3, p, label, span {
    color: white !important;
}

#main-title {
    text-align: center;
    color: #7dd3fc !important;
}

#subtitle {
    text-align: center;
    color: #cbd5e1 !important;
}

.warning-box {
    background: rgba(255, 193, 7, 0.12);
    border: 1px solid rgba(255, 193, 7, 0.35);
    padding: 12px;
    border-radius: 12px;
    color: #fde68a;
}

.footer-note {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 13px;
}
"""


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        """
# 🫁 Lung Cancer AI Detection System
### CT Scan Classification with EfficientNetB0 and Grad-CAM
""",
        elem_id="main-title"
    )

    gr.Markdown(
        """
Upload a lung CT scan image. The AI predicts one of four classes and shows the image regions that influenced its decision.
""",
        elem_id="subtitle"
    )

    gr.Markdown(
        """
⚠️ **Research use only:** This tool is not intended for clinical diagnosis or medical decision-making.
""",
        elem_classes=["warning-box"]
    )

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.Image(type="pil", label="1. Upload CT scan image")
            analyze_btn = gr.Button("Analyze CT Scan", variant="primary")
            result = gr.Markdown(label="Prediction Result")

        with gr.Column(scale=2):
            with gr.Row():
                original_out = gr.Image(label="2. Original CT Image")
                overlay_out = gr.Image(label="3. AI Focus Overlay")

            with gr.Row():
                heatmap_out = gr.Image(label="4. Grad-CAM Heatmap")
                prob_chart_out = gr.Image(label="5. Class Probability Chart")

    explanation = gr.Markdown(label="AI Explanation")

    gr.Markdown(
        """
**Model:** EfficientNetB0 transfer learning  
**Explainability method:** Grad-CAM  
**Classes:** Adenocarcinoma, large cell carcinoma, normal, squamous cell carcinoma
""",
        elem_classes=["footer-note"]
    )

    analyze_btn.click(
        fn=predict_ct,
        inputs=upload,
        outputs=[
            result,
            original_out,
            heatmap_out,
            overlay_out,
            prob_chart_out,
            explanation
        ]
    )

demo.launch()
