import torch.nn as nn

def tcnn2():
    tcnn2 =nn.Sequential(
        nn.Conv2d(1,96,kernel_size=3,padding=1), #96x576x576 #256x256
        nn.MaxPool2d(2,2), #96x288x288 #128x128

        nn.Conv2d(96,256,kernel_size=3,padding=1), #256x288x288 #128x128
        nn.MaxPool2d(2,2), #256x144x144 #64x64

        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1), #256x144x144 #64x64

        nn.AvgPool2d((64, 64)),
        nn.Flatten(),

        nn.Linear(256,4096),
        nn.ReLU(),

        nn.Linear(4096,4096),
        nn.ReLU(),

        nn.Linear(4096,28))
    return tcnn2

def tcnn3():
    tcnn3 =nn.Sequential(
	#CONV-1
	nn.Conv2d(1,96,kernel_size=11,stride=4, padding=2), # 1x227x227 -> 96x55x55
	nn.ReLU(inplace=True),
	nn.MaxPool2d((3,3), stride=2), # 96x55x55 -> 96x27x27

	#CONV-2
	nn.Conv2d(96,256,kernel_size=5,padding=2), # 96x27x27 -> 256x27x27
	nn.ReLU(inplace=True),
	nn.MaxPool2d((3,3), stride=2), # 256x27x27 -> 256x13x13

	#CONV-3
	nn.Conv2d(256,384,kernel_size=3,padding=1), # 256x13x13 -> 384x13x13
	nn.ReLU(inplace=True),

	#EnergyPooling
	nn.AvgPool2d((15, 15)),
	nn.Flatten(),
	nn.ReLU(),

	#FC1
	nn.Linear(384,4096),
	nn.ReLU(),

	#FC2
	nn.Linear(4096,4096),
	nn.ReLU(),

	#nn.Linear(4096,28)
	nn.Linear(4096,28),
	#nn.Softmax()
	    )
    return tcnn3
