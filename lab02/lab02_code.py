import numpy as np


class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name

        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []

        # Reset the node-state (not the graph topology)
        self.reset()

    def reset(self):
        # Incomming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}

        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])

    def add_neighbour(self, nb):
        self.neighbours.append(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')

    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')

    def receive_msg(self, other, msg):
        # Store the incomming message, replacing previous messages from the same node
        self.in_msgs[other] = msg

        # TODO: add pending messages
        # self.pending.update(...)

    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing.
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states

        # Call the base-class constructor
        super(Variable, self).__init__(name)

    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0

    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate observed an latent
        # variables when sending messages.
        self.observed_state[:] = 1.0

    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)

    def marginal(self, Z=None):
        """
        Compute the marginal distribution of this Variable.
        It is assumed that message passing has completed when this function is called.
        Args:
            Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        Returns: Z. Either equal to the input Z, or computed (if Z=None was passed).
        """
        prob = 1
        for fac in self.neighbours:
            msg = self.in_msgs[fac]
            prob *= msg
        if not Z:
            Z = sum(prob)
        marg = prob/Z
        return marg, Z

    def send_sp_msg(self, other):
        """
        Variable -> Factor message for sum-product
        """
        nbs = filter(not nb == other for nb in self.neighbours)
        if len(nbs) == 0:
            other.receive_msg(self, np.array([1]*self.num_states))
        else:
            pass # TODO implement


    def send_ms_msg(self, other):
        """
        Variable -> Factor message for max-sum
        """
        msg = 0
        nbs = filter(not nb == other for nb in self.neighbours)
        if len(nbs) == 0:
            #leaf node
            other.receive_msg(self, np.array([0]*self.num_states))
        else:
            for f in nbs:
                msg += f.in_msgs[self]
            other.receive_msg(self, msg)

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)

        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'

        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f

    def send_sp_msg(self, other):
        """
        Factor -> Variable message for sum-product.
        """
        vectors = []
        for neighbour in filter(not n == other for n in self.neighbours):
            vectors.append(neighbour.marginal())
        if len(vectors) == 0:
            msg = self.f
        else:
            msg = np.product.reduce(np.ix_(*vectors))
        other.send(self, msg)

    def send_ms_msg(self, other):
        """
        Factor -> Variable message for max-sum.
        """
        # TODO: implement Factor -> Variable message for max-sum
        nbs = filter(not nb == other for nb in self.neighbours)
        if len(nbs) == 0:
            #leaf node
            max_msg = np.log(self.f)
        else:
            #TODO figure this out
            #Try out all possible values for x_1,...,x_n
            #Take the max
            max_msg = np.array([-1000]*self.num_states)
            msg = np.log(self.f)
            for x in nbs:
                msg += self.in_msgs[x] 
        other.send(self, max_msg)
        pass


def instantiate_network():
    VARIABLE_NAMES = ['Influenza', 'Smokes', 'SoreThroat', 'Fever',
                      'Bronchitis', 'Coughing', 'Wheezing']
    variables = {name: Variable(name, 2) for name in VARIABLE_NAMES}

    f = {}
    #p(Influenza)=0.05
    f['f_I'] = np.array([0.05, 0.95])

    #p(Smokes)=0.2
    f['f_S'] = np.array([0.2, 0.8])

    #p(SoreThroat=1|Influenza=1)=0.3
    #p(SoreThroat=1|Influenza=0)=0.001
    f['f_ISt'] = np.array([[0.3, 0.7],
                           [0.001, 0.999]])

    #p(Bronchitis=1|Influenza=1,Smokes=1)=0.99
    #p(Bronchitis=1|Influenza=1,Smokes=0)=0.9
    #p(Bronchitis=1|Influenza=0,Smokes=1)=0.7
    #p(Bronchitis=1|Influenza=0,Smokes=0)=0.0001
    f['f_ISB'] = np.array([[[0.99, 0.9],
                            [0.7, 0.001]],
                           [[0.01, 0.1],
                            [0.3, 0.999]]])

    #p(Fever=1|Influenza=1)=0.9
    #p(Fever=1|Influenza=0)=0.05
    f['f_IF'] = np.array([[0.9, 0.1],
                          [0.05, 0.95]])

    #p(Wheezing=1|Bronchitis=1)=0.6
    #p(Wheezing=1|Bronchitis=0)=0.001
    f['f_BW'] = np.array([[0.6, 0.4],
                          [0.001, 0.999]])

    #p(Coughing=1|Bronchitis=1)=0.8
    #p(Coughing=1|Bronchitis=0)=0.07
    f['f_BC'] = np.array([[0.8, 0.2],
                          [0.07, 0.93]])

    FACTORS = [('f_I', [variables['Influenza']]),
               ('f_S', [variables['Smokes']]),
               ('f_ISt', [variables['Influenza'],
                          variables['SoreThroat']]),
               ('f_ISB', [variables['Smokes'],
                          variables['Influenza'],
                          variables['Bronchitis']]),
               ('f_IF', [variables['Influenza'],
                         variables['Fever']]),
               ('f_BW', [variables['Bronchitis'],
                         variables['Wheezing']]),
               ('f_BC', [variables['Bronchitis'],
                         variables['Coughing']])]

    factors = {name: Factor(name, f[name], n) for name, n in FACTORS}

instantiate_network()

