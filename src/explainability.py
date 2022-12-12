from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
import torch
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap

class Explainer:
    def __init__(self):
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                            [(0, '#ffffff'),
                                                            (0.25, '#000000'),
                                                            (1, '#000000')], N=256)
    # TODO: Blur upscaling
    def shap(self, model, image, pred_label_idx):
        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        torch.manual_seed(0)
        np.random.seed(0)

        input = transform_normalize(image)
        input = input.unsqueeze(0)

        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([input * 0, input * 1])

        attributions_gs = gradient_shap.attribute(input,
                                                n_samples=50,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "absolute_value"],
                                            cmap=self.default_cmap,
                                            show_colorbar=True)
        
    def occlusion(self, model, image, pred_label_idx):
        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(input,
                                            strides = (3, 8, 8),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)
        
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )