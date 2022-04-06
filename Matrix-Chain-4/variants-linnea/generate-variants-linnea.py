from linnea.algebra.expression import Matrix, Vector, Equal, Times, Inverse, Transpose
from linnea.algebra.equations import Equations
from linnea.algebra.properties import Property

def matrix_chain_equation():
    n = 10
    m = 100
    k = 80
    l = 150
    q = 120

    A = Matrix("A", (n, m))
    A.set_property(Property.FULL_RANK)

    B = Matrix("B", (m, k))
    B.set_property(Property.FULL_RANK)

    C = Matrix("C", (k, l))
    C.set_property(Property.FULL_RANK)
    
    D = Matrix("D", (l, q))
    D.set_property(Property.FULL_RANK)

    Y = Matrix("Y", (n, q))

    # Y = ABCD
    equations = Equations(Equal(Y, Times(A,B,C,D)))

    return equations



if __name__ == "__main__":

    import linnea.config

    linnea.config.set_output_code_path(".")
    linnea.config.init()

    from linnea.algorithm_generation.graph.search_graph import SearchGraph

    #from input1 import equations

    # import linnea.examples.examples
    # equations = linnea.examples.examples.Example001().eqns

    equations = matrix_chain_equation()
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

