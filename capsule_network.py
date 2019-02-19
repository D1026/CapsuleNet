"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Ivan_Duan @ AntFin.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()

# --- for checking ---
# self.capsule_layer = capsule_network.CapsuleLayer(num_labels, 1, config.hidden_size, output_atoms,
#                                                   num_iterations=num_iterations)
# --- ---


class Original_CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, leaky=False, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(Original_CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        self.leaky = leaky

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
            self.route_biases = nn.Parameter(torch.randn(num_capsules, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def leaky_route(self, x):
        leak = Variable(torch.zeros((*x.size()))).to(x.device.type)
        leak = leak.sum(dim=0, keepdim=True)
        leak_x = torch.cat((leak, x), 0)
        leak_probs = softmax(leak_x, dim=0)
        return leak_probs[1:, :, :, :, :]

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]  # shape [j, b ,i, 1, o]
            p_size = [*priors.size()]
            logits = Variable(torch.zeros(p_size)).to(priors.device.type)
            for i in range(self.num_iterations):
                # leaky
                if self.leaky:
                    probs = self.leaky_route(logits)
                else:
                    probs = softmax(logits, dim=0)
                # probs = softmax(logits, dim=2)
                # outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                # Wx plus b，reduce sum over input_atoms dimmension.
                # preactivate = (probs * priors).sum(dim=2, keepdim=True) + self.route_biases[:, None, None, None, :]
                preactivate = (probs * priors).sum(dim=2, keepdim=True) + self.route_biases[:, None, None, None, :] # [j, b, 1, 1, o]
                outputs = self.squash(preactivate)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, input_atoms, output_atoms, num_routing=3, leaky=False, kernel_size=None, stride=None,
                 ):
        super(CapsuleLayer, self).__init__()
        self.input_shape = (input_dim, input_atoms)  # omit batch dim
        self.output_shape = (output_dim, output_atoms)
        self.num_routing = num_routing
        self.leaky = leaky
        self.weights = nn.Parameter(torch.randn(input_dim, input_atoms, output_dim * output_atoms))
        self.biases = nn.Parameter(torch.randn(output_dim, output_atoms))


    def _squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def _leaky_route(self, x, output_dim):
        leak = torch.zeros(x.shape).to(x.device.type)
        leak = leak.sum(dim=2, keepdim=True)
        leak_x = torch.cat((leak, x), 2)
        leaky_routing = softmax(leak_x, dim=2)
        return leaky_routing[:, :, :output_dim]


    def forward(self, x):
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.output_shape[0]*self.output_shape[1])  # [b, i, i_o, j*j_o]
        votes = torch.sum(x * self.weights, dim=2) # [b, i, j*j_o]
        votes_reshaped = torch.reshape(votes,
                                    [-1, self.input_shape[0], self.output_shape[0], self.output_shape[1]])
                                    # [b, i, j, j_o]
        # routing loop
        logits = torch.zeros(x.shape[0], self.input_shape[0], self.output_shape[0]).to(x.device.type) # [b, i, j]
        for i in range(self.num_routing):
            if self.leaky:
                route = self._leaky_route(logits, self.output_shape[0])
            else:
                route = softmax(logits, dim=2)
            route = route.unsqueeze(-1) # [b, i, j, 1]
            preactivate_unrolled = route * votes_reshaped   # [b, i, j, j_o]
            s = preactivate_unrolled.sum(1, keepdim=True) + self.biases # [b, 1, j, j_o]
            v = self._squash(s)

            distances = (votes_reshaped * v).sum(dim=3) # [b, i, j]
            logits = logits + distances

        # return v
        return torch.transpose(v, 0, 2)  # just adapt original API

class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


class CapsuleClassification(nn.Module):
    def __init__(self):
        super(CapsuleClassification, self).__init__()

    # x, the capsule layer output
    def forward(self, x):
        x = x.squeeze().transpose(0, 1)
        classes_probs = (x ** 2).sum(dim=-1) ** 0.5
        classes_ids = F.softmax(classes_probs, dim=-1)
        return classes_probs, classes_ids


class CapsuleClassificationLoss(nn.Module):
    def __init__(self, label_dim):
        super(CapsuleClassificationLoss, self).__init__()
        self.label_format = nn.Parameter(torch.arange(label_dim).reshape(1, label_dim), requires_grad=False)

    def forward(self, labels, classes):
        one_hot_target = (labels.unsqueeze(-1) == self.label_format).float()
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = one_hot_target * left + 0.5 * (1. - one_hot_target) * right
        margin_loss = margin_loss.sum()

        return margin_loss
