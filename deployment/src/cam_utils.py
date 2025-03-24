import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def look(src):
    plt.imshow(src)
    plt.show()


class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.reshape_transform = reshape_transform

        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation
                )
            )
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, obs, goal):
        self.gradients = []
        self.activations = []
        return self.model(obs, goal)

    def release(self):
        for handle in self.handles:
            handle.remove()


class get_loss(torch.nn.Module):
    def __init__(self, beta=1):
        super(get_loss, self).__init__()
        self.beta = beta

    def forward(self, mu, logvar):
        # recon_x, x: bs, t, 2
        # mu, logvar: bs*t, latent_size
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kldivergence


class GradCAM:
    def __init__(self, activations_and_grads, target_size):
        self.activations_and_grads = activations_and_grads
        self.target_size = target_size

    @staticmethod
    def get_cam_weights(activations, grads):  # GAP

        # gradcam ++
        # grads_power_2 = grads ** 2
        # grads_power_3 = grads_power_2 * grads
        # # Equation 19 in https://arxiv.org/abs/1710.11063
        # sum_activations = np.sum(activations, axis=(2, 3))
        # eps = 0.000001
        # aij = grads_power_2 / (2 * grads_power_2 +
        #                        sum_activations[:, :, None, None] * grads_power_3 + eps)
        # # Now bring back the ReLU from eq.7 in the paper,
        # # And zero out aijs where the activations are 0
        # aij = np.where(grads != 0, aij, 0)
        #
        # weights = np.maximum(grads, 0) * aij
        # weights = np.sum(weights, axis=(2, 3))
        # return weights

        # gradcam
        return np.mean(grads, axis=(2,3))

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def get_cam_image(self, activations, grads):
        # gradcam, gradcam++
        weights = self.get_cam_weights(activations, grads)  # GAP
        weighted_activations = weights[:, :, None, None] * activations
        # without eigen smooth
        cam = weighted_activations.sum(axis=1)
        # with eigen smooth
        # cam = self.get_2d_projection(weighted_activations)

        ##hirescam
        # elementwise_activations = grads * activations
        # cam = elementwise_activations.sum(axis=1)

        ##eigengradcam
        # cam = self.get_2d_projection(grads * activations)

        return cam

    def get_2d_projection(self, activation_batch):
        # with eigen smooth
        activation_batch[np.isnan(activation_batch)] = 0
        projections = []
        for activations in activation_batch:
            reshaped_activations = (activations).reshape(activations.shape[0], -1).transpose()
            # Centering before the SVD seems to be important here,
            # Otherwise the image returned is negative
            reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activations.shape[1:])
            projections.append(projection)
        cam = np.float32(projections)
        return cam

    @staticmethod
    def scale_cam_img(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv.resize(img, target_size)  # cv2.resize(src, (width, height))
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self):
        target_size = self.target_size
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [a.cpu().data.numpy() for a in self.activations_and_grads.gradients]
        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # ReLU
            scaled = self.scale_cam_img(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_layer):
        cam_per_layer = np.concatenate(cam_per_layer, axis=1)
        cam_per_layer = np.maximum(cam_per_layer, 0)
        result = np.mean(cam_per_layer, axis=1)
        return self.scale_cam_img(result)

    def __call__(self, activations_and_grads):
        self.activations_and_grads = activations_and_grads

        cam_per_layer = self.compute_cam_per_layer()
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv.COLORMAP_JET):
    gray = np.uint8(255 * mask)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # count_anomaly = np.count_nonzero(thresh == 255)
    count_anomaly_left = np.count_nonzero(thresh[:, 0:28] == 255)
    count_anomaly_mid = np.count_nonzero(thresh[:, 28:46] == 255)
    count_anomaly_right = np.count_nonzero(thresh[:, 46:85] == 255)
    heatmap = cv.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    img = np.uint8(255 * cam)
    # draw boundingbox
    # cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     x, y, w, h = cv.boundingRect(c)
    #     cv.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
    return img, count_anomaly_left, count_anomaly_mid, count_anomaly_right
