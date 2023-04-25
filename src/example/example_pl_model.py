from thunder.pl_model.abstract_pl_model import AbstractPlModule


class ExamplePlModel(AbstractPlModule):

    def forward(self, x, y):
        output = self.model(x)
        return output
