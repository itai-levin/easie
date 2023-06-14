import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from rdkit import Chem

def draw_bipartite_route(graph):
    pos = graphviz_layout(graph, prog="dot")
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=500,
        font_size=2,
        arrowsize=10,
        width=0.3,
        node_shape="s",
        node_color="k",
    )
    ax = plt.gca()
    fig = plt.gcf()
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    imsize = 0.075  # this is the image size
    for n in graph.nodes():
        if ">" in n:
            continue
        (x, y) = pos[n]
        xx, yy = trans((x, y))  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
        mol = Chem.MolFromSmiles(n)
        img = Chem.Draw.MolToImage(mol)
        a.imshow(img)
        a.set_aspect("equal")
        a.axis("off")
    plt.show()
