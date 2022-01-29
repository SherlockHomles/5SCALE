from Tree import Tree, ConeTree, SpheroidTree
from math import pi, exp
from scipy.stats import poisson


class Domain(object):
    '''
    the class contains necessary attributes of a domain or pixel comprised of quadrats that grows trees
    Attributes:
        tree: it must be classes inherited from Tree
        area: domain area
        n_tree: total number of trees in the domain
        n_quadrat: number of quadrats in the domain
        Fr: if fr==1, there is no overlap at nadir
        m2: the mean size of clusters, if 0, poisson tree distribution, otherwise, neyman distribution (Eq.5)
        distribution: tree distribution in the domain, determined by m2. Poissson distribution if m2 =0, otherwise,
                        Neyman distribution
        Omega_stand: clumping index of stand, it equals to Omega_E
        Px: an array of poisson or neyman distribution, depending on m2 or distribution type

    Methods:
        _poisson: return an array of poisson distribution
        _neyman: return an array of neyman distribution

    '''

    def __init__(self, tree: Tree, area: float, n_tree: int, n_quadrat: float, Fr: float = 1, m2: float = 0):
        self.tree = tree
        self.area = area
        self.n_tree = n_tree
        self.n_quadrat = n_quadrat
        self.Fr = Fr
        self.m2 = m2
        Omega_stand = self.Omega_stand
        if 0 < Omega_stand < 1:
            Omega_E = Omega_stand + (1 - Omega_stand) / 2
            self.tree.Omega_E = Omega_E

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, value: Tree):
        self._check_inputs(value, 'tree')
        self._tree = value

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value: float):
        if value <= 0:
            raise ValueError('domain area must be larger than 0')
        self._check_inputs(value, 'area')
        self._area = value

    @property
    def n_tree(self):
        return self._n_tree

    @n_tree.setter
    def n_tree(self, value: int):
        if value < 0:
            raise ValueError('number of trees cannot smaller than 0')
        self._check_inputs(value, 'n_tree')
        self._n_tree = value

    @property
    def n_quadrat(self):
        return self._n_quadrat

    @n_quadrat.setter
    def n_quadrat(self, value: int):
        if value < 0:
            raise ValueError('number of quadrats cannot be smaller than 0')
        self._n_quadrat = value

    @property
    def Fr(self):
        return self._Fr

    @Fr.setter
    def Fr(self, value: float):
        if value < 0:
            raise ValueError('Fr cannot be smaller than 0')
        self._check_inputs(value, 'Fr')
        self._Fr = value

    @property
    def m2(self):
        return self._m2

    @m2.setter
    def m2(self, value: float):
        if value < 0:
            raise ValueError('m2 cannot be smaller than 0')
        self._check_inputs(value, 'm2')
        self._m2 = value

    @property
    def distribution(self):
        if self.m2 == 0:
            dist = 'POISSON'
        else:
            dist = 'NEYMAN'
        return dist

    @property
    def Omega_stand(self):
        Omega = self.tree.Omega_E
        return Omega

    @property
    def Px(self):
        # returns an array of poisson or neyman distribution depending on distribution type
        if self.distribution == 'POISSON':
            px = self._poisson()
        else:
            px = self._neyman()
        return px

    @property
    def Px_poisson(self):
        # returns an array of poisson distribution
        return self._poisson()

    @property
    def MAX_TREE_PER_QUADRAT(self):
        return 350  # max number of trees in a quadrat, this is a final constant

    # this function is used to ensure the total surface of tree crowns is smaller than domain area in case Fr>0.5 and
    # m2 ==0
    def _check_inputs(self, value, type_of_value):
        if type_of_value == 'tree':
            if not isinstance(value, SpheroidTree) and not isinstance(value, ConeTree):
                raise TypeError('tree type must be either ConeTree or SpheroidTree')
            if hasattr(self, '_area') and hasattr(self, '_n_tree') and hasattr(self, '_Fr') and hasattr(self, '_m2'):
                R = value.R
                n_tree = self.n_tree
                area = self.area
                Fr = self.Fr
                distribution = self.distribution
            else:
                return
        elif type_of_value == 'area':
            if hasattr(self, '_tree') and hasattr(self, '_n_tree') and hasattr(self, '_Fr') and hasattr(self, '_m2'):
                R = self.tree.R
                n_tree = self.n_tree
                area = value
                Fr = self.Fr
                distribution = self.distribution
            else:
                return
        elif type_of_value == 'n_tree':
            if hasattr(self, '_tree') and hasattr(self, '_area') and hasattr(self, '_Fr') and hasattr(self, '_m2'):
                R = self.tree.R
                n_tree = value
                area = self.area
                Fr = self.Fr
                distribution = self.distribution
            else:
                return
        elif type_of_value == 'Fr':
            if hasattr(self, '_tree') and hasattr(self, '_area') and hasattr(self, '_n_tree') and hasattr(self, '_m2'):
                R = self.tree.R
                n_tree = self.n_tree
                area = self.area
                Fr = value
                distribution = self.distribution
            else:
                return
        else:
            if hasattr(self, '_tree') and hasattr(self, '_area') and hasattr(self, '_n_tree') and hasattr(self, '_Fr'):
                R = self.tree.R
                n_tree = self.n_tree
                area = self.area
                Fr = self.Fr
                distribution = 'POISSON'
            else:
                return
        crown_area = pi * R ** 2 * n_tree
        if crown_area > area and Fr > 0.5 and distribution == 'POISSON':
            raise ValueError(
                'domain area is smaller than the total surface of the crowns, please enter a reasonable number of '
                'trees')

    def _poisson(self):  # return an array of poisson parameters
        px = []
        mu = self.n_tree / self.n_quadrat
        for i in range(self.MAX_TREE_PER_QUADRAT):
            p = poisson.pmf(i, mu)
            px.append(p)
        px_total = poisson.cdf(self.MAX_TREE_PER_QUADRAT, mu)
        if px_total < 0.95:
            Error_info = 'sum of all tree distribution probabilities is less than 0.95 (sum = {:8.5f})'.format(px_total)
            raise ValueError(Error_info)
        else:
            px = [x / px_total for x in px]
        return px

    def _neyman(self):  # return an array of neyman parameters
        '''computes the neyman distribution used to simulate clumping of trees'''
        d = self.n_tree
        n = self.n_quadrat
        m2 = self.m2
        m1 = d / (n * m2)
        px = [0 for _ in range(self.MAX_TREE_PER_QUADRAT)]
        px[0] = exp(-m1 * (1. - exp(-m2)))
        px_tot = px[0]
        for k in range(1, self.MAX_TREE_PER_QUADRAT):
            px[k] = 0
            if k > d / n and px[k - 1] < 0.000001:
                px[k] = 0
            else:
                for t in range(k):
                    pxx = m1 * m2 * exp(-m2) / k * px[k - t - 1]
                    for i in range(1, t + 1):
                        pxx = pxx * m2 / i
                    px[k] = px[k] + pxx
            px_tot = px_tot + px[k]
        if px_tot < 0.98:
            Error_info = 'sum of all tree distribution probabilities is less than 0.98 (sum = {:8.5f})'.format(px_tot)
            raise ValueError(Error_info)
        px = [x / px_tot for x in px]
        return px