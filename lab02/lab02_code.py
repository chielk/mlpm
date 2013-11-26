import numpy as np

DEBUG = True

from pprint import pprint

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
        if DEBUG:
            print msg, 'received by', self.name, 'from', other.name
        self.in_msgs[other] = msg

        # add pending messages
        self.pending.update(set(self.neighbours) - set([other]))

    def sent_msg(self, other):
        self.pending.discard(other)

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
        return {'marginal': marg, 'Z': Z}

    def send_sp_msg(self, other):
        """
        Variable -> Factor message for sum-product
        """
        nbs = filter(lambda nb: not nb == other, self.neighbours)
        if len(nbs) == 0:
            msg = np.array([1] * self.num_states)
        else:
            vectors = [self.in_msgs[nb] for nb in nbs]
            msg = np.multiply.reduce(vectors)
        other.receive_msg(self, msg)
        self.sent_msg(other)  # For pending messages


    def send_ms_msg(self, other):
        # TODO: implement Variable -> Factor message for max-sum
        pass


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
        nbs = filter(lambda nb: not nb == other, self.neighbours)
        if len(nbs) == 0:
            msg = self.f
        else:
            vectors = [self.in_msgs[nb] for nb in nbs]
            #msg = numpy.matrix(self.f) * numpy.matrix(vectors).T
            mm = reduce(np.multiply, np.ix_(*vectors))
            other_i = self.neighbours.index(other)
            f_axes = filter(lambda i: not i == other_i, range(len(self.neighbours)))
            msg = np.tensordot(self.f, mm, axes=(f_axes, range(mm.ndim)))
        other.receive_msg(self, msg)
        self.sent_msg(other)  # For pending messages

    def send_ms_msg(self, other):
        # TODO: implement Factor -> Variable message for max-sum
        pass

def instantiate_network():
    VARIABLES = ['Influenza', 'Smokes', 'SoreThroat', 'Fever',
                 'Bronchitis', 'Coughing', 'Wheezing']
    v_ = {name: Variable(name, 2) for name in VARIABLES}

    f_ = {}
    #p(Influenza)=0.05
    f_['f_I'] = np.array([0.05, 0.95])

    #p(Smokes)=0.2
    f_['f_S'] = np.array([0.2, 0.8])

    #p(SoreThroat=1|Influenza=1)=0.3
    #p(SoreThroat=1|Influenza=0)=0.001
    f_['f_ISt'] = np.array([[0.3, 0.7],
                           [0.001, 0.999]])

    #p(Bronchitis=1|Influenza=1,Smokes=1)=0.99
    #p(Bronchitis=1|Influenza=1,Smokes=0)=0.9
    #p(Bronchitis=1|Influenza=0,Smokes=1)=0.7
    #p(Bronchitis=1|Influenza=0,Smokes=0)=0.0001
    f_['f_ISB'] = np.array([[[0.99, 0.9],
                            [0.7, 0.001]],
                           [[0.01, 0.1],
                            [0.3, 0.999]]])

    #p(Fever=1|Influenza=1)=0.9
    #p(Fever=1|Influenza=0)=0.05
    f_['f_IF'] = np.array([[0.9, 0.1],
                          [0.05, 0.95]])

    #p(Wheezing=1|Bronchitis=1)=0.6
    #p(Wheezing=1|Bronchitis=0)=0.001
    f_['f_BW'] = np.array([[0.6, 0.4],
                          [0.001, 0.999]])

    #p(Coughing=1|Bronchitis=1)=0.8
    #p(Coughing=1|Bronchitis=0)=0.07
    f_['f_BC'] = np.array([[0.8, 0.2],
                          [0.07, 0.93]])

    FACTORS = [('f_I', [v_['Influenza']]),
               ('f_S', [v_['Smokes']]),
               ('f_ISt', [v_['Influenza'],
                          v_['SoreThroat']]),
               ('f_ISB', [v_['Smokes'],
                          v_['Influenza'],
                          v_['Bronchitis']]),
               ('f_IF', [v_['Influenza'],
                         v_['Fever']]),
               ('f_BW', [v_['Bronchitis'],
                         v_['Wheezing']]),
               ('f_BC', [v_['Bronchitis'],
                         v_['Coughing']])]

    f_ = {name: Factor(name, f_[name], n) for name, n in FACTORS}
    return f_, v_

f_, v_ = instantiate_network()

nodes = [f_['f_S'],
         f_['f_I'],
         v_['SoreThroat'],
         v_['Fever'],
         v_['Coughing'],
         v_['Wheezing'],
         f_['f_ISt'],
         f_['f_IF'],
         f_['f_BC'],
         f_['f_BW'],
         v_['Smokes'],
         v_['Bronchitis'],
         v_['Influenza'],
         f_['f_ISB']]

def sum_product(node_list):
    # Forward
    for i, node in enumerate(node_list):
        for neighbour in filter(lambda n: n not in node_list[:i],
                                node.neighbours):
            print node.name, '>', neighbour.name
            node.send_sp_msg(neighbour)
            print
    # Set pending
    for node in node_list:
        if len(node.neighbours) == 1:
            node.pending.add(node.neighbours[0])

    print
    print '===== set pending nodes ====='
    print
    print

    # Back
    reverse_nodes = list(reversed(node_list))
    for i, node in enumerate(reverse_nodes):
        for neighbour in filter(lambda n: n not in reverse_nodes[:i],
                                node.neighbours):
            print node.name, '>', neighbour.name
            node.send_sp_msg(neighbour)
            print

sum_product(nodes)
