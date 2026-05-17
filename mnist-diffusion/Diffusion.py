class Diffusion:
    def __init__(self):
        params = locals().pop("self")
        self.__set_parameters(params=params)

    def __set_parameters(self, *, params: dict):
        for key, value in params:
            setattr(self, key, value)
