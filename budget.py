from impedance_budget_toolbox.element import Element
from typing import List


class Budget:
    """
    Suggestion for structure of Budget class
    """
    def __init__(self, elements: List[Element] = None):
        assert elements, "Budget object needs to be initialized with at least one Element"
        self._elements = elements
        self._model = sum(elements)
        self.x_impedance = lambda x: 0
        self.y_impedance = lambda x: 0
        self.z_impedance = lambda x: 0
        self.x_wake = lambda x: 0
        self.y_wake = lambda x: 0
        self.z_wake = lambda x: 0
        self._calculate_impedance_wake()

    def _calculate_impedance_wake(self):
        for c in self._model.components:
            if c.plane == 'x':
                self.x_impedance = lambda x: self.x_impedance(x) + c.impedance(x)
                self.x_wake = lambda x: self.x_wake(x) + c.wake(x)
            elif c.plane == 'y':
                self.y_impedance = lambda x: self.y_impedance(x) + c.impedance(x)
                self.y_wake = lambda x: self.y_wake(x) + c.wake(x)
            elif c.plane == 'z':
                self.z_impedance = lambda x: self.z_impedance(x) + c.impedance(x)
                self.z_wake = lambda x: self.z_wake(x) + c.wake(x)

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements: List[Element]):
        self._elements = elements
        self._model = sum(elements)
        self._calculate_impedance_wake()

    @property
    def model(self):
        return self._model

    def add_element(self, element: Element):
        self._elements.append(element)
        self._model += element
