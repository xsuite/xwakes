from pywib.element import Element

from typing import List


class Budget:
    """
    Suggestion for structure of Budget class
    """
    def __init__(self, elements: List[Element] = None):
        assert elements, "Budget object needs to be initialized with at least one Element"
        self._elements = elements
        self._model = sum(elements)



    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, elements: List[Element]):
        self._elements = elements
        self._model = sum(elements)

    @property
    def model(self):
        return self._model

    def add_element(self, element: Element):
        self._elements.append(element)
        self._model += element
