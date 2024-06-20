from .element import Element

from typing import List, Optional, Tuple


class Model:
    """
    Suggestion for structure of Model class
    """
    def __init__(self, elements: List[Element] = None, lumped_betas: Optional[Tuple[float, float]] = None):
        assert elements, "Model object needs to be initialized with at least one Element"
        if lumped_betas is not None:
            elements = [element.changed_betas(*lumped_betas) for element in elements]

        self.__elements = elements
        self.__lumped_betas = lumped_betas

    @property
    def elements(self):
        return self.__elements

    @property
    def total(self):
        return sum(self.__elements)

    def append_element(self, element: Element):
        if self.__lumped_betas is not None:
            element = element.changed_betas(*self.__lumped_betas)
        self.__elements.append(element)
