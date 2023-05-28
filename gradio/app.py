import PIL
from captum.attr import GradientShap, Occlusion, LayerGradCam, LayerAttribution, IntegratedGradients
from captum.attr import visualization as viz
import torch
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import torch.nn.functional as F
import gradio as gr
from torchvision.models import resnet50
import torch.nn as nn
import torch
import numpy as np

MODEL_NAMES = ["imagenet_finetuned", "not_finetuned", "masked_model", "rects_model"]
IMGNET_MEAN = np.array([0.485, 0.456, 0.406])
IMGNET_STD = np.array([0.229, 0.224, 0.225])

class Explainer:
    def __init__(self, model, img, class_names, model_name):
        self.model_name = model_name
        self.model = model
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                [(0, '#ffffff'),
                                                (0.25, '#000000'),
                                                (1, '#000000')], N=256)
        self.jet_cmap = cm.get_cmap("cubehelix")
        self.class_names = class_names

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        transform_normalize = transforms.Normalize(
            mean=IMGNET_MEAN,
            std=IMGNET_MEAN
        )

        self.transformed_img = transform(img)

         # find all pixels that are exactly equal to 0
        if self.model_name == "rects_model":
            mask = np.all(self.transformed_img.cpu().detach().numpy() == 0, axis=0)

        self.input = transform_normalize(self.transformed_img)
        self.input = self.input.unsqueeze(0)

        if self.model_name == "rects_model":
            # replace pixels in mask with 0
            self.input[:, :, mask == 1] = 0

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
                                            cmap=self.jet_cmap,
                                            show_colorbar=True)
        fig.suptitle("SHAP | " + self.fig_title, fontsize=12)
        return self.convert_fig_to_pil(fig)

    def occlusion(self, stride, sliding_window):
        occlusion = Occlusion(self.model)

        x = np.zeros((3, 224, 224))
        base = -np.divide(IMGNET_MEAN, IMGNET_STD)
        x[0, :, :] = base[0]
        x[1, :, :] = base[1]
        x[2, :, :] = base[2]

        attributions_occ = occlusion.attribute(self.input,
                                               target=self.pred_label_idx,
                                               strides=(3, int(stride), int(stride)),
                                               sliding_window_shapes=(3, int(sliding_window), int(sliding_window)),
                                               baselines=torch.tensor(x, dtype=torch.float32))

        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map", "heat_map", "masked_image"],
                                            ["all", "positive", "negative", "positive"],
                                            show_colorbar=True,
                                            titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                            fig_size=(18, 6),
                                            cmap=cm.get_cmap("magma"),
                                            )
        fig.suptitle("Occlusion | " + self.fig_title, fontsize=12)
        return self.convert_fig_to_pil(fig)
    
    def gradcam(self):
        layer_gradcam = LayerGradCam(self.model, self.model.layer4[-1].conv3)
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
                                            fig_size=(18, 6),
                                            cmap=cm.get_cmap("magma"),
                                        )
        fig.suptitle("GradCAM layer4[-1].conv3 | " + self.fig_title, fontsize=12)
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
    if len(model_name) == 0:
        return "Please select a model"

    if not model_name in preloaded_models:
        preloaded_models[model_name] = create_model_from_checkpoint(model_name + ".h5")
    
    explainer = Explainer(preloaded_models[model_name], img, labels, model_name)
    
    return [
            explainer.confidences,
            "True label: " + true_label,
            explainer.shap(shap_samples, shap_stdevs),
            explainer.occlusion(occlusion_stride, occlusion_window),
            explainer.gradcam(),
        ]

shap_samples = 70
shap_stdevs = 0.0001
occlusion_stride = 16
occlusion_window = 24

vis_hypers = [ shap_samples, shap_stdevs, occlusion_stride, occlusion_window ]

examples = [
    ["examples/original/benign/benign (25).png", "imagenet_finetuned", "benign", *vis_hypers],
    ["examples/original/malignant/malignant (149).png", "imagenet_finetuned", "malignant", *vis_hypers],
    ["examples/original/normal/normal (101).png", "imagenet_finetuned", "normal", *vis_hypers],

    ["examples/masked/benign/benign (10)_inv_mult.png", "masked_model", "benign", *vis_hypers],
    ["examples/masked/malignant/malignant (23)_inv_mult.png", "masked_model", "malignant", *vis_hypers],
    ["examples/masked/normal/normal (4)_inv_mult.png", "masked_model", "normal", *vis_hypers], 

    ["examples/rects/benign/benign (10)_inv_mult.png", "rects_model", "benign", *vis_hypers],
    ["examples/rects/malignant/malignant (15)_inv_mult.png", "rects_model", "malignant", *vis_hypers],
    ["examples/rects/normal/normal (39)_inv_mult_rect.png", "rects_model", "normal", *vis_hypers], 
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