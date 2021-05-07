# imports
from scipy.stats import multiscale_graphcorr as MGC
import numpy as np
import matplotlib.pyplot as plt
from inspect import getmembers
import warnings

import sys

sys.path.append('//home/azureuser/MGC_classifier/source')
import shape_functions


class function_generator:
    def __init__(self, sample_size=32, output_size=32):
        
        # the size of the samples that go through MGC
        self.SIZE = sample_size
        # the size of the samples that actually come out
        # of the generator through interpolating
        self.output_size = output_size

        self.operations = [
            "multiply",
            "divide",
            "add",
            "subtract",
        ]

        self.hyppo_functions = [
            "linear",
            "exponential",
            "cubic",
            "joint_normal",
            "step",
            "quadratic",
            "w_shaped",
            "spiral",
            "logarithmic",
            "fourth_root",
            "sin_four_pi",
            "sin_sixteen_pi",
            "square",
            "two_parabolas",
            "circle",
            "ellipse",
            "diamond",
            "multiplicative_noise",
            "multimodal_independence",
        ]

        # two term operations
        self.two_term_operations = ["multiply", "divide"]

        # TODO adding and subtracting new terms
        self.add_term_operations = ["add", "subtract"]

        # check if all operations are contained in the other sets
        assert set(self.two_term_operations).union(
            set(self.add_term_operations)
        ) == set(self.operations)

        self.sample_size = sample_size

        self.operation_scale_dict = {
            "arccos": (-0.99, 0.99),
            "arccosh": (1.01, 10),
            "arcsin": (-0.99, 0.99),
            "arcsinh": (-np.pi, np.pi),
            "arctan": (-3, 3),
            "arctan2": ((-0.99, 0.99), (-0.99, 0.99)),
            "arctanh": (-0.99, 0.99),
            "heaviside": ((-0.99, 0.99), (-0.99, 0.99)),
            "log": (0.01, 3),
            "log10": (0.01, 3),
            "log1p": (0.01, 3),
            "log2": (0.01, 3),
            "multiply": ((-1, 1), (-1, 1)),
            "sin": (-np.pi, np.pi),
            "sinh": (-3, 3),
            "sqrt": (0, 1),
            "square": (-1, 1),
            "tan": (-np.pi / 2 + 0.01, np.pi / 2 - 0.01),
            "tanh": (-np.pi, np.pi),
            "divide": ((-2, 2), (0.01, 2)),
            "subtract": (-1, 1),
            "add": (-1, 1),
        }

        self.samples_dict = {}

    def yield_sample(self, a, b, p):
        """calls the other methods to yield samples lazily"""
        while True:

            # this controls how many successive operations are going
            n_functions = np.random.choice([1, 2, 3])

            # get a random set of functions but make sure they are mixable
            functions = None
            while functions is None or \
                (set(functions).intersection([
                    "joint_normal", 
                    "spiral",
                    "uncorrelated_bernoulli",
                    "circle",
                    "ellipse",
                    "multimodal_independence",
                ]) and len(functions)>1):
                functions = np.random.choice(self.hyppo_functions, size = n_functions)

            # start a dictionary to store values
            funct_dict = {}

            # get the values for these functions
            for i, function in enumerate(functions):
                # get function and noise
                funct = getattr(shape_functions, function)
                noise = bool(np.random.choice([True, False], p=[0.1, 0.9]))
                # handle passing x to ensure that xs are the same
                if i==0:
                    x = None
                if function in ["multiplicative_noise", "multimodal_independence"]:
                    x, y = funct(self.sample_size, 1)

                elif function in ["linear","exponential","cubic", "step","quadratic", 
                                "w_shaped", "fourth_root", "sin_four_pi", 
                                "sin_sixteen_pi","square", 
                                "diamond"]:
                    x, y = funct(self.SIZE,1, noise=noise, low=min(a,b), high=max(a,b), x=x)

                elif function in ["spiral", "circle", "ellipse"]:
                    x, y = funct(self.SIZE,1, noise=noise, low=min(a,b), high=max(a,b))

                elif function =="two_parabolas":
                    x, y = funct(self.SIZE,1, noise=noise, low=min(a,b), high=max(a,b), 
                                prob=p, x=x)
  
                elif function == 'uncorrelated_bernoulli':
                    x, y = funct(self.SIZE,1, noise=noise, prob=p)

                elif function in ["joint_normal"]:
                    x, y = funct(self.SIZE,1, noise=noise)
                
                elif function in ["logarithmic"]:
                    x, y = funct(self.SIZE,1, noise=noise, x=x)

                elif function == 'multimodal_independence':
                    x, y = funct(self.SIZE,1, prob=p, sep1=a, sep2=b)
                
                funct_dict[function] = {
                    'x':x,
                    'y':y
                }

            # store the operations for targets
            n_operations = n_functions-1
            order_of_operations = np.random.choice(self.operations, size = n_operations)

            # initialize a list for the terms
            terms = []
            x, y = None, None
            for i, function in enumerate(functions):

                if (x is None) or (y is None):
                    x = funct_dict[function]['x']
                    y = funct_dict[function]['y']
                    
                else:
                    # get the actual function with this operator
                    y = y/np.nanmax(np.abs(y))
                    operator = order_of_operations[i-1]
                    funct = getattr(np, operator)
                    
                    # get the new values
                    y_new = funct_dict[function]['y']/np.max(funct_dict[function]['y'])

                    # new value for y
                    y = funct(y,y_new)
                    
            inf_mask = ~(np.abs(y) == np.inf)
            nan_mask = ~np.isnan(y)
            x, y = x[nan_mask & inf_mask], y[nan_mask & inf_mask]

            yield (x, y), (*functions, *order_of_operations)

    def generate_MGC_maps(self, n_samples, generator_parameters):

        a, b, p = generator_parameters
        # temporary
        a,b,p = -1,1,0.5
        # instantiate the generator
        gen = self.yield_sample(a,b,p)

        labels = []
        # generate the data
        for i in range(n_samples):
            print(f"\r{i} {len(self.samples_dict)}", end="")
            (x, y), label = next(gen)

            labels.append(label)

            try:
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, _, MGC_DICT = MGC(x, y, reps=0, workers=-1)
                    fig, axs = plt.subplots(1,2)
                    axs[0].imshow(MGC_DICT["mgc_map"])
                    axs[0].set_title(f'{label[:(len(label)-1)]}\n'+
                    f'{label[(len(label)-1):]}')
                    axs[0].invert_yaxis()
                    axs[1].plot(x, y, 'ro', markersize=2)
                    plt.pause(0.001)

                self.samples_dict[i] = (MGC_DICT["mgc_map"], (x, y), label)


            except IndexError:
                pass


print("a")
class_instance = function_generator(32)
a_test,b_test,p_test = -1,1,0.5
generator = class_instance.yield_sample(a_test,b_test,p_test)
class_instance.generate_MGC_maps(100, (a_test,b_test,p_test))

# should be around 12 gb of data ofr 100,000 samples
