import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from model import Net

# Import ART
from art.estimators.classification import PyTorchClassifier

"""
    White-Box Attacks
"""
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import AutoAttack
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import ShadowAttack
from art.attacks.evasion import wasserstein
from art.attacks.evasion import BrendelBethgeAttack
from art.attacks.evasion import targeted_universal_perturbation
from art.attacks.evasion import HighConfidenceLowUncertainty
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import CarliniLInfMethod


"""
    Black-Box Attacks
"""
from art.utils import to_categorical
from art.estimators.classification.blackbox import BlackBoxClassifier, BlackBoxClassifierNeuralNetwork
from art.attacks.evasion import SquareAttack
from art.attacks.evasion import SimBA
from art.attacks.evasion import ThresholdAttack
from art.attacks.evasion import PixelAttack

"""
    Patch
"""
from art.attacks.evasion import AdversarialPatch


def load_data(dataset_path):
    img = Image.open(dataset_path)

    img_ndarray = np.asarray(img, dtype='float64') / 256


    faces = np.empty((400, 57 * 47))

    for row in range(20):
        for column in range(20):
            faces[20 * row + column] = np.ndarray.flatten(
                img_ndarray[row * 57: (row + 1) * 57, column * 47 : (column + 1) * 47]
            )

    label = np.zeros((400, 40))
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1


    train_data = np.empty((320, 57 * 47))
    train_label = np.zeros((320, 40))
    vaild_data = np.empty((40, 57 * 47))
    vaild_label = np.zeros((40, 40))
    test_data = np.empty((40, 57 * 47))
    test_label = np.zeros((40, 40))

    for i in range(40):
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]

        vaild_data[i] = faces[i * 10 + 8]
        vaild_label[i] = label[i * 10 + 8]

        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    vaild_data = vaild_data.astype('float32')
    test_data = test_data.astype('float32')

    return [
        (train_data, train_label),
        (vaild_data, vaild_label),
        (test_data, test_label)
    ]


epsilons = [.05,.06, .065, .070, .075, .08, .085,.1]
epsilons = np.array(epsilons)



pretrained_model = "face.pth"
dataset_path = "olivettifaces.gif"

use_cuda = torch.cuda.is_available()
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


model = Net(1, 40).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

print("The model is:")
print(model)
# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
print("Model is now in the evaluation mode.")



dataset = load_data(dataset_path)
#print("data")
#train_set_x = dataset[0][0]
#train_set_y = dataset[0][1]
#valid_set_x = dataset[1][0]
#valid_set_y = dataset[1][1]
test_set_x = dataset[2][0]
test_set_y = dataset[2][1]

data, target = torch.from_numpy(test_set_x), torch.from_numpy(test_set_y)
data = torch.reshape(data, [-1, 1, 57, 47])

print("The size of the input is:")
print(data.shape)

data, target = data.to(device), target.to(device)

"""
    White-Box Classifier
"""
classifier = PyTorchClassifier(
            model=model,
            input_shape=(data.shape),
            nb_classes=40,
            loss=nn.CrossEntropyLoss(),
            device_type="cpu"
        )

original_predictions = classifier.predict(data)

accuracy = np.sum(np.argmax(original_predictions, axis=1) == np.argmax(test_set_y, axis=1)) / test_set_y.shape[0]
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


# Generate adversarial test examples

"""
    White-Box Attacks
"""
# FGSM
"""attacker = FastGradientMethod(
    estimator=classifier,
    eps=0.1
    )"""

"""
# Auto-PGD
attacker = AutoProjectedGradientDescent(
    estimator=classifier,
    loss_type="cross_entropy"
)"""

"""
# Auto Attack
attacker = AutoAttack(
    estimator=classifier
)
"""

"""# Shadow Attack
attacker = ShadowAttack(
    estimator=classifier,
    batch_size=1
)
"""

""" # Wasserstein---Works but not good
    # Be careful with the parameters!
attacker = wasserstein.Wasserstein(
    estimator=classifier,
    regularization=1000.0,
    eps=0.1,
    max_iter=100,
    conjugate_sinkhorn_max_iter=100,
    projected_sinkhorn_max_iter=100
)"""
"""
# Brendel and Bethge Attack
attacker = BrendelBethgeAttack(
    estimator=classifier
)
"""
"""# Targeted Universal
attacker_param = {
        "estimator": classifier,
        "norm": "inf", 
        "eps": 0.3,
        "eps_step": 0.1,
        "targeted": False,
        "num_random_init": 0,
        "batch_size": 32,
        "minimal": False
        }

attacker = targeted_universal_perturbation.TargetedUniversalPerturbation(
    classifier=classifier,
    attacker_params=attacker_param
)
"""
"""# HCLU
attacker = HighConfidenceLowUncertainty(
    classifier=classifier
)"""

""" # PGD It works!
    # Mind the parameters!
attacker = ProjectedGradientDescentPyTorch(
    estimator=classifier,
    eps=0.1
)"""

""" # C&W Attack(L_inf) It works!
    # Mind the parameters
attacker = CarliniLInfMethod(
    classifier=classifier,
    confidence=0.0
)
"""
"""
# C&W Attack(L_2) It works!
# Mind the parameters
attacker = CarliniL2Method(
    classifier=classifier,
    confidence=0.0
)
"""

"""
    Black-Box Classifier
"""

def predict(data):
    original_predictions = classifier.predict(data)
    return original_predictions

BlackBox = BlackBoxClassifierNeuralNetwork(
    predict=predict,
    input_shape=data.shape,
    nb_classes=40,
    clip_values=(0, 1)
)

"""
    Black-Box
"""

# BlackBox_predictions = BlackBox.predict(data)

"""# Square Attack
attacker = SquareAttack(
    estimator=BlackBox
)"""


"""# SimBA Hardly works
attacker = SimBA(
    classifier=BlackBox,
    attack="px",
    max_iter=300,
    epsilon=0.1
)"""
# ThresholdAttack
attacker = ThresholdAttack(
    classifier=BlackBox,
    verbose=True
)

"""
# PixelAttack
attacker = PixelAttack(
    classifier=BlackBox,
    verbose=True
)
"""

"""
    Patch
"""

"""# For Targeted Universal Adversarial Attack
label_perturbed = np.eye(test_set_y.shape[0])
np.random.shuffle(label_perturbed)

x_test_adv = attacker.generate(
    x=data,
    y=label_perturbed
    )
"""

# For others
x_test_adv = attacker.generate(
    x=data
    )

# Evaluate the performance
perturbed_predictions = classifier.predict(x_test_adv)

accuracy = np.sum(np.argmax(perturbed_predictions, axis=1) == np.argmax(test_set_y, axis=1)) / test_set_y.shape[0]
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))






#-------------------------------


accuracies = []
examples = []


"""# Run test for each epsilon
for eps in epsilons:
    acc = test(model, device, dataset, eps, attack='fgsm')
    #acc, ex = test(model, device, dataset, eps, attack='fgsm')
    #acc, ex = test(model, device, test_loader, eps, attack='illc')
    accuracies.append(acc)
    #examples.append(ex)
"""
"""accuracies = np.array(accuracies)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
#plt.yticks(np.arange(0, 1.1, step=0.1))
#plt.xticks(np.arange(0, 100, step=10))
plt.xlim(0, 0.1)
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig("accuracy_illc3040_mul.png")
plt.show()"""



"""
# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.savefig("samples_illc3040_mul.png")
plt.show()

"""