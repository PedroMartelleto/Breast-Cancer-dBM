from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, NoiseTunnel, GradientShap
from captum.attr import visualization as viz
import numpy as np
import torch
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
import datasets
import torch.nn.functional as F
import os

class Explainer:
    def __init__(self, model, device, ds):
        self.model = model
        self.device = device
        self.ds = ds
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                            [(0, '#ffffff'),
                                                            (0.25, '#000000'),
                                                            (1, '#000000')], N=256)

    def prepare(self, img):
        torch.cuda.empty_cache()

        self.model = self.model.to(self.device)
        self.model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        transform_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transformed_img = transform(img)

        input = transform_normalize(transformed_img)
        input = input.unsqueeze(0)
        input = input.to(self.device)

        output = self.model(input)
        output = F.softmax(output, dim=1)
        pred_score, pred_label_idx = torch.topk(output, 1)
        pred_label = self.ds.class_names[pred_label_idx]
        fig_title = 'Predicted: ' + pred_label + ' (' + str(round(pred_score.squeeze().item(), 2)) + ')'
        print(fig_title)

        return input, transformed_img, pred_label_idx, pred_label, pred_score, fig_title

    def shap(self, img, dst_file):
        input, transformed_img, pred_label_idx, pred_label, pred_score, fig_title = self.prepare(img)

        gradient_shap = GradientShap(self.model)
        rand_img_dist = torch.cat([input * 0, input * 1])
        attributions_gs = gradient_shap.attribute(input, n_samples=50, stdevs=0.0001, baselines=rand_img_dist, target=pred_label_idx)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "absolute_value"],
                                            cmap=self.default_cmap,
                                            show_colorbar=True)
        fig.suptitle(fig_title, fontsize=12)
        fig.savefig(dst_file)

    def occlusion(self, img, dst_file):
        input, transformed_img, pred_label_idx, pred_label, pred_score, fig_title = self.prepare(img)

        occlusion = Occlusion(self.model)
        attributions_occ = occlusion.attribute(input,
                                               strides = (3, 8, 8),
                                               target=pred_label_idx,
                                               sliding_window_shapes=(3,15, 15),
                                               baselines=0)
        
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              show_colorbar=True,
                                              outlier_perc=2)
        fig.suptitle(fig_title, fontsize=12)
        fig.savefig(dst_file)

    def noise_tunnel(self, img, dst_file):
        input, transformed_img, pred_label_idx, pred_label, pred_score, fig_title = self.prepare(img)
        
        integrated_gradients = IntegratedGradients(self.model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        fig, _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["original_image", "heat_map"],
                                            ["all", "positive"],
                                            cmap=self.default_cmap,
                                            show_colorbar=True)
        fig.suptitle(fig_title, fontsize=12)
        fig.savefig(dst_file)

    def gradcam(self, img, dst_file):
        input, transformed_img, pred_label_idx, pred_label, pred_score, fig_title = self.prepare(img)

        layer_gradcam = LayerGradCam(self.model, self.model.layer3[1].conv2)
        attributions_lgc = layer_gradcam.attribute(input, target=pred_label_idx)

        fig, _ = viz.visualize_image_attr_multiple(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                                    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                    ["original_image", "heat_map"],
                                                    sign="all",
                                                    title="Layer 3 Block 1 Conv 2")
        fig.suptitle(fig_title, fontsize=12)
        fig.savefig(dst_file)