import PIL
from captum.attr import GradientShap, Occlusion, LayerGradCam, LayerAttribution, IntegratedGradients
from captum.attr import visualization as viz
import torch
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
import gradio as gr
from torchvision.models import resnet50
import torch.nn as nn
import torch
import numpy as np

MODEL_NAMES = ["imagenet_finetuned", "not_finetuned", "masked_model"]

class Explainer:
    def __init__(self, model, img, class_names):
        self.model = model
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                [(0, '#ffffff'),
                                                (0.25, '#000000'),
                                                (1, '#000000')], N=256)
        self.class_names = class_names

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.transformed_img = transform(img)

        self.input = transform_normalize(self.transformed_img)
        self.input = self.input.unsqueeze(0)

        with torch.no_grad():
            self.output = self.model(self.input)
            self.output = F.softmax(self.output, dim=1)

        self.confidences = {class_names[i]: float(self.output[0, i]) for i in range(3)}

        self.pred_score, self.pred_label_idx = torch.topk(self.output, 1)
        self.pred_label = self.class_names[self.pred_label_idx]
        self.fig_title = 'Predicted: ' + self.pred_label + ' (' + str(round(self.pred_score.squeeze().item(), 2)) + ')'

    def convert_fig_to_pil(self, fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return PIL.Image.fromarray(data)

    def shap(self, n_samples, stdevs):
        gradient_shap = GradientShap(self.model)
        rand_img_dist = torch.cat([self.input * 0, self.input * 1])
        attributions_gs = gradient_shap.attribute(self.input, n_samples=int(n_samples), stdevs=stdevs, baselines=rand_img_dist, target=self.pred_label_idx)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "absolute_value"],
                                            cmap=self.default_cmap,
                                            show_colorbar=True)
        fig.suptitle("SHAP | " + self.fig_title, fontsize=12)
        return self.convert_fig_to_pil(fig)

    def occlusion(self, stride, sliding_window):
        occlusion = Occlusion(self.model)

        attributions_occ = occlusion.attribute(self.input,
                                               target=self.pred_label_idx,
                                               strides=(3, int(stride), int(stride)),
                                               sliding_window_shapes=(3, int(sliding_window), int(sliding_window)),
                                               baselines=0)

        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map", "heat_map", "masked_image"],
                                            ["all", "positive", "negative", "positive"],
                                            show_colorbar=True,
                                            titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                            fig_size=(18, 6)
                                            )
        fig.suptitle("Occlusion | " + self.fig_title, fontsize=12)
        return self.convert_fig_to_pil(fig)
    
    def gradcam(self):
        layer_gradcam = LayerGradCam(self.model, self.model.layer3[1].conv2)
        attributions_lgc = layer_gradcam.attribute(self.input, target=self.pred_label_idx)

        #_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
        #                            sign="all",
        #                            title="Layer 3 Block 1 Conv 2")
        upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, self.input.shape[2:])

        fig, _ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                            self.transformed_img.permute(1,2,0).numpy(),
                                            ["original_image","blended_heat_map","masked_image"],
                                            ["all","positive","positive"],
                                            show_colorbar=True,
                                            titles=["Original", "Positive Attribution", "Masked"],
                                            fig_size=(18, 6))
        fig.suptitle("GradCAM layer3[1].conv2 | " + self.fig_title, fontsize=12)
        return self.convert_fig_to_pil(fig)

def create_model_from_checkpoint(model_name):
    # Loads a model from a checkpoint
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    model.eval()
    return model

preloaded_models = {}
labels = [ "benign", "malignant", "normal" ]

def predict(img, model_name, true_label, shap_samples, shap_stdevs, occlusion_stride, occlusion_window):
    if not model_name in preloaded_models:
        preloaded_models[model_name] = create_model_from_checkpoint(model_name + ".h5")
    
    explainer = Explainer(preloaded_models[model_name], img, labels)
    
    return [
            explainer.confidences,
            "True label: " + true_label,
            explainer.shap(shap_samples, shap_stdevs),
            explainer.occlusion(occlusion_stride, occlusion_window),
            explainer.gradcam()
        ]

examples = [
    ["examples/original/benign/benign (52).png", "imagenet_finetuned", "benign", 50, 0.0001, 8, 15],
    ["examples/original/benign/benign (243).png", "imagenet_finetuned", "benign", 50, 0.0001, 8, 15],
    ["examples/original/malignant/malignant (149).png", "imagenet_finetuned", "malignant", 50, 0.0001, 8, 15],
    ["examples/original/malignant/malignant (201).png", "imagenet_finetuned", "malignant", 50, 0.0001, 8, 15],
    ["examples/original/normal/normal (100).png", "imagenet_finetuned", "normal", 50, 0.0001, 8, 15], 
    ["examples/original/normal/normal (101).png", "imagenet_finetuned", "normal", 50, 0.0001, 8, 15],

    ["examples/masked/benign/benign (10)_inv_mult.png", "masked_model", "benign", 50, 0.0001, 8, 15],
    ["examples/masked/benign/benign (93)_inv_mult.png", "masked_model", "benign", 50, 0.0001, 8, 15],
    ["examples/masked/malignant/malignant (23)_inv_mult.png", "masked_model", "malignant", 50, 0.0001, 8, 15],
    ["examples/masked/malignant/malignant (59)_inv_mult.png", "masked_model", "malignant", 50, 0.0001, 8, 15],
    ["examples/masked/normal/normal (4)_inv_mult.png", "masked_model", "normal", 50, 0.0001, 8, 15], 
    ["examples/masked/normal/normal (95)_inv_mult.png", "masked_model", "normal", 50, 0.0001, 8, 15],  
]

ui = gr.Interface(fn=predict, 
                inputs=[
                    gr.Image(type="pil"),
                    gr.Dropdown(MODEL_NAMES, default="imagenet_finetuned"),
                    gr.Dropdown(["benign", "malignant", "normal"], default="normal"),
                    gr.Slider(minimum=10, maximum=100, default=50, label="SHAP Samples", step=1),
                    gr.Slider(minimum=0.0001, maximum=0.01, default=0.0001, label="SHAP Stdevs", step=0.0001),
                    gr.Slider(minimum=4, maximum=80, default=8, label="Occlusion Stride", step=1),
                    gr.Slider(minimum=4, maximum=80, default=15, label="Occlusion Window", step=1),
                ],
                outputs=[
                         gr.Label(num_top_classes=3),
                         gr.Label(),
                         gr.Image(type="pil"),
                         gr.Image(type="pil"),
                         gr.Image(type="pil")
                ],
                examples=examples)

ui.launch(share=False, server_name="0.0.0.0")