"""NN layer that outputs monomials

This layer outputs all possible monomials up to the given degree from
the list of inputs.

"""

import torch
from torch import nn
import math

class MonomialLayer(nn.Module):
    """Outputs all possible monomials up to given degree from inpupts.

    The basic idea is to add the number 1 to the list of inputs and
    then create every possible monomial of the given degree from
    factors of the inputs and one.  The presence of one in the list
    generates the monomials with degree less than the given degree.
    See the math of multisets for more information.  (Wikipedia has a
    good entry on this.

    Attributes:
       n_inputs: The number of inputs this layer expects
       degree: The maximum degree of the momomial
       n_outputs: The number of monomials output

    """
    def __init__(self, n_inputs: int, degree: int):
        super(MonomialLayer,self).__init__()

        self.n_inputs = n_inputs
        self.degree = degree
        # No point in keeping the constant term
        self.n_outputs = int(math.factorial(n_inputs+degree) /
                          math.factorial(n_inputs) /
                          math.factorial(degree)) - 1

        # Now, let's build an array of the indices of the inputs that
        # need to be combined for each monomial
        self.m_ind = torch.zeros(self.n_outputs,degree,dtype=torch.int32)
        curr_ind = torch.zeros(self.degree, dtype=torch.int32)

        for row in range(self.n_outputs):
            # Calculate the values for this row
            for col in range(self.degree-1,0,-1):
                if curr_ind[col-1] > curr_ind[col]:
                    curr_ind[col]+=1
                    break
                else:
                    curr_ind[col]=0
            else:
                curr_ind[0]+=1
                curr_ind[1:]=0 # Broadcasts!

            # Set the indices for this row
            self.m_ind[row,:] =  curr_ind

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.n_inputs:
            raise IndexError(f'Expecting {self.n_inputs} inputs, got {x.shape[-1]}')
        x = torch.cat((torch.ones(x.shape[:-1]+(1,)),x),axis=-1)
        return torch.prod(torch.index_select(x,-1,self.m_ind.flatten())
                          .reshape(x.shape[:-1]+self.m_ind.shape),axis=-1)
