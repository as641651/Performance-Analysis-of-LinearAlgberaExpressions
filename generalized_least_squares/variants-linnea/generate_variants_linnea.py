from linnea.algebra.expression import Matrix, Vector, Equal, Times, Inverse, Transpose
from linnea.algebra.equations import Equations
from linnea.algebra.properties import Property


def generalized_least_squares():
    n = 500
    m = 2500

    b = Vector("b", (n, 1))

    X = Matrix("X", (m, n))
    X.set_property(Property.FULL_RANK)

    M = Matrix("M", (m, m))
    M.set_property(Property.SPD)

    y = Vector("y", (m, 1))

    # b = (X.T*M.inv*X).inv*X.T*M.inv*y
    equations = Equations(Equal(b, Times(Inverse(Times(Transpose(X), Inverse(M), X)), Transpose(X), Inverse(M), y)))

    return equations


if __name__ == "__main__":

    import linnea.config

    linnea.config.set_output_code_path(".")
    linnea.config.init()

    from linnea.algorithm_generation.graph.search_graph import SearchGraph

    #from input1 import equations

    # import linnea.examples.examples
    # equations = linnea.examples.examples.Example001().eqns

    equations = generalized_least_squares()
    graph = SearchGraph(equations)
    graph.generate(time_limit=60,
                   merging=True,
                   dead_ends=True,
                   pruning_factor=100.0)

    graph.write_output(code=True,
                       generation_steps=True,
                       output_name="variants",
                       experiment_code=False,
                       algorithms_limit=100,
                       graph=True,
                       no_duplicates=True)

